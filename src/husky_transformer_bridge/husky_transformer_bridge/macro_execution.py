"""Training macro semantics driven by confirmed primitive execution.

Reference: archived code_evaluation/thesis/envs/goal_selection_wrapper.py and
astar_planner.py. Directions: right=0, down=1, left=2, up=3. Right turns count
as steps; forward into a wall is a completed blocked primitive, not translation.
Physical execution and its timeouts are the caller's responsibility.
"""
import math

GOALS = ((1, 1), (15, 1), (1, 9), (15, 9))
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


class MacroExecution:
    def __init__(self, planner, open_cells, lambda_values, max_steps=1500):
        if type(max_steps) is not int or max_steps <= 0 or max_steps % 5:
            raise ValueError('Episode limit must be a positive multiple of five')
        if len(lambda_values) != 4 or not all(math.isfinite(x) and 0 <= x <= 10 for x in lambda_values):
            raise ValueError('Four raw lambda values in [0,10] required')
        self.planner = planner
        self.open_cells = set(open_cells)
        self.lam = tuple(lambda_values)
        self.max_steps = max_steps
        self.episode_steps = 0
        self.astar_events = 0
        self.active = False
        self.pending = None

    def begin(self, action, cell, direction):
        if self.active or self.episode_steps >= self.max_steps:
            raise RuntimeError('Macro already active or episode ended')
        if type(action) is not int or action not in range(4):
            raise ValueError('Invalid selected goal')
        if tuple(cell) not in self.open_cells or type(direction) is not int or direction not in range(4):
            raise ValueError('Invalid primitive state')
        self.action, self.cell, self.direction = action, tuple(cell), direction
        self.path = []
        self.steps = 0
        self.reward = 0.0
        self.trace = []
        self.active = True

    def next_primitive(self):
        if not self.active or self.pending is not None:
            raise RuntimeError('No macro ready, or primitive still awaiting confirmation')
        if not self.path:
            path = self.planner(self.cell, GOALS[self.action])
            self.astar_events += 1
            self.path = [tuple(c) for c in path] if path else []
        primitive = 2
        if len(self.path) > 1:
            target = self.path[1]
            delta = (target[0]-self.cell[0], target[1]-self.cell[1])
            if delta not in DIRECTIONS:
                raise ValueError('Planner supplied a nonadjacent primitive')
            if self.direction != DIRECTIONS.index(delta):
                primitive = 1
            else:
                self.path.pop(0)
        new_direction = (self.direction + 1) % 4 if primitive == 1 else self.direction
        new_cell = self.cell
        if primitive == 2:
            dx, dy = DIRECTIONS[self.direction]
            candidate = (self.cell[0]+dx, self.cell[1]+dy)
            if candidate in self.open_cells:
                new_cell = candidate
        self.pending = dict(primitive_action=primitive, start=self.cell,
                            end=new_cell, direction=new_direction,
                            blocked=primitive == 2 and new_cell == self.cell)
        return dict(self.pending)

    def confirm(self, actual_cell, actual_direction):
        if self.pending is None:
            raise RuntimeError('No primitive awaiting confirmation')
        p = self.pending
        if tuple(actual_cell) != p['end'] or actual_direction != p['direction']:
            raise ValueError('Observed primitive result does not match commanded primitive')
        self.cell, self.direction = tuple(actual_cell), actual_direction
        reward = 0.0
        if (self.episode_steps + 1) % 5 == 0 and self.cell in GOALS:
            reward = self.lam[GOALS.index(self.cell)] * 0.99 ** max(0, self.astar_events - 1)
        self.reward += 0.99 ** self.steps * reward
        self.steps += 1
        self.episode_steps += 1
        self.trace.append(dict(p, primitive_reward=reward, episode_step=self.episode_steps))
        self.pending = None
        done = self.steps == 5 or self.episode_steps >= self.max_steps
        if done:
            self.active = False
        return done

    def feedback(self):
        if self.active or self.steps != 5 or self.pending is not None:
            raise RuntimeError('Cannot report an incomplete macro as completed')
        return dict(action=self.action, reward=self.reward, terminated=False,
                    truncated=self.episode_steps >= self.max_steps,
                    primitive_steps=self.steps, episode_primitive_steps=self.episode_steps,
                    astar_events=self.astar_events, end_cell=list(self.cell),
                    end_direction=self.direction, trace=self.trace)
