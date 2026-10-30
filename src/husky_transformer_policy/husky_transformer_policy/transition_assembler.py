"""Recorded macro feedback adapter. No invented real-robot rewards."""


class RecordedTransitions:
    def __init__(self, data, count):
        import numpy as np
        self.data = data
        self.count = count
        required = ('obs_position', 'obs_lambda', 'action', 'macro_reward', 'terminated',
                    'truncated', 'episode_decision_id', 'decision_id')
        if count < 1 or any(k not in data or len(data[k]) < count for k in required):
            raise ValueError('Incomplete recording')
        if data['obs_position'].shape[1:] != (2,) or data['obs_lambda'].shape[1:] != (4,):
            raise ValueError('Invalid recorded observation shapes')
        for key in ('obs_position', 'obs_lambda', 'macro_reward'):
            if not np.isfinite(data[key][:count]).all(): raise ValueError('Nonfinite recording')
        pos = data['obs_position'][:count]
        if not (pos == np.floor(pos)).all() or not ((pos >= 0) & (pos <= [16, 10])).all():
            raise ValueError('Invalid raw grid positions')
        lam = data['obs_lambda'][:count]
        if not ((lam >= 0) & (lam <= 10)).all() or not (lam == lam[0]).all():
            raise ValueError('Replay must contain one fixed-lambda task')
        acts = data['action'][:count]
        if not ((acts >= 0) & (acts < 4) & (acts == np.floor(acts))).all():
            raise ValueError('Invalid recorded actions')
        if data['terminated'].dtype != np.bool_ or data['truncated'].dtype != np.bool_:
            raise ValueError('Recorded end flags must be boolean')
        if int(data['decision_id'][0]) != 0 or int(data['episode_decision_id'][0]) != 0:
            raise ValueError('Replay starts from true history/episode start')
        if count > 1:
            if not (np.diff(data['decision_id'][:count]) == 1).all():
                raise ValueError('Noncontiguous decision recording')
            ends = data['terminated'][:count-1] | data['truncated'][:count-1]
            expected = np.where(ends, 0, data['episode_decision_id'][:count-1]+1)
            if not (expected == data['episode_decision_id'][1:count]).all():
                raise ValueError('Episode/feedback boundary mismatch')

    @property
    def lambda_values(self):
        return self.data['obs_lambda'][0].astype(float).tolist()

    def observation(self, i, epoch):
        return dict(epoch=epoch, history_version=i, state_seq=i,
                    agent_pos=self.data['obs_position'][i].astype(int).tolist())

    def feedback(self, i, epoch, ticket):
        return dict(epoch=epoch, history_version=i, state_seq=i, ticket=ticket,
                    action=int(self.data['action'][i]), reward=float(self.data['macro_reward'][i]),
                    terminated=bool(self.data['terminated'][i]), truncated=bool(self.data['truncated'][i]))
