"""Isolated candidate-to-path bench. All metric coordinates use a synthetic grid."""
import ast, json, os, sys, time
from pathlib import Path
if (os.environ.get('ROS_DOMAIN_ID'), os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE')) != ('87', 'LOCALHOST'):
    raise RuntimeError('Use run_planner.sh in isolated ROS domain 87')
if not __debug__:
    raise RuntimeError('Run with Python assertions enabled')
sys.path.insert(0, os.environ['TRAINING_WHEEL'])
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement
from husky_transformer_policy.protocol import request
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav2_msgs.action import ComputePathToPose
from nav2_msgs.srv import GetCostmap
from nav_msgs.msg import OccupancyGrid, Path as RosPath
from geometry_msgs.msg import PoseStamped
sock=os.environ['BENCH_SOCKET']
task=request(sock,'init_task',lambda_values=[7,2,2,2])
start=(4,3)
decision=request(sock,'infer',epoch=task['epoch'],history_version=0,state_seq=1,agent_pos=list(start))
goal=tuple(decision['goal_grid_xy'])
rclpy.init()
node=rclpy.create_node('candidate_path_bench')
def pose(x,y):
    p=PoseStamped()
    p.header.frame_id='map'
    p.pose.orientation.w=1.
    p.pose.position.x=(x+.5)*0.5
    p.pose.position.y=(y+.5)*0.5
    return p
source=Path(os.environ['HUSKY_WS'])/'src/husky_rl/husky_rl/astar_node.py'
cells=next(ast.literal_eval(n.value) for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign) and n.targets[0].id=='OPEN_CELLS')
grid=Grid(matrix=[[int((x,y) in cells) for x in range(17)] for y in range(11)])
route,_=AStarFinder(diagonal_movement=DiagonalMovement.never).find_path(grid.node(*start),grid.node(*goal),grid)
pts=[(p.x,p.y) for p in route]
assert pts[0]==start and pts[-1]==goal and all(p in cells for p in pts)
metric=RosPath()
metric.header.frame_id='map'
metric.header.stamp=node.get_clock().now().to_msg()
metric.poses=[pose(*p) for p in pts]
qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
out=node.create_publisher(RosPath,'/transformer/shadow/training_path',qos)
out.publish(metric)
maps=node.create_publisher(OccupancyGrid,'/map',qos)
planner=ActionClient(node,ComputePathToPose,'/compute_path_to_pose')
cost=node.create_client(GetCostmap,'/global_costmap/get_costmap')
assert planner.wait_for_server(timeout_sec=8) and cost.wait_for_service(timeout_sec=8)
def wait(f):
    rclpy.spin_until_future_complete(node,f,timeout_sec=8)
    assert f.done()
    return f.result()
assert goal==(1,1), goal
summary=dict(mode='synthetic_bench',start=start,goal=goal,model=decision['model_hash'],ticket=decision['ticket'],training_cells=pts,training_meters=[(p.pose.position.x,p.pose.position.y) for p in metric.poses],cases={})
for case in ('open','detour','blocked'):
    m=OccupancyGrid()
    m.header.frame_id='map'
    m.header.stamp=node.get_clock().now().to_msg()
    m.info.width,m.info.height=85,55
    m.info.resolution=.1
    m.info.origin.orientation.w=1.
    m.data=[100 if x==15 and (case=='blocked' or case=='detour' and not 30<=y<=42) else 0 for y in range(55) for x in range(85)]
    maps.publish(m)
    deadline=time.monotonic()+12
    while time.monotonic()<deadline:
        actual=wait(cost.call_async(GetCostmap.Request())).map
        if actual.metadata.size_x==85 and actual.metadata.size_y==55 and len(actual.data)==len(m.data) and all((v>=253)==(w==100) for v,w in zip(actual.data,m.data)):break
        time.sleep(.15)
    else: raise RuntimeError('costmap did not match '+case)
    req=ComputePathToPose.Goal()
    req.start,req.goal,req.use_start,req.planner_id=pose(*start),pose(*goal),True,'GridBased'
    handle=wait(planner.send_goal_async(req))
    assert handle.accepted
    reply=wait(handle.get_result_async())
    path=reply.result.path
    coords=[(p.pose.position.x,p.pose.position.y) for p in path.poses]
    if case=='blocked':
        assert reply.status==6 and not coords and reply.result.error_code==ComputePathToPose.Result.NO_VALID_PATH,(reply.status,reply.result.error_code)
    else:
        assert reply.status==4 and coords and path.header.frame_id=='map'
        assert all(m.data[int(y/.1)*85+int(x/.1)]==0 for x,y in coords)
        assert abs(coords[-1][0]-req.goal.pose.position.x)<.11 and abs(coords[-1][1]-req.goal.pose.position.y)<.11
        if case=='detour': assert max(y for x,y in coords)>=3
    summary['cases'][case]=dict(status=reply.status,error=reply.result.error_code,points=len(coords))
summary['history_version']=request(sock,'status')['history_version']
assert summary['history_version']==0
Path('result.json').write_text(json.dumps(summary,indent=2))
print(summary)
node.destroy_node()
rclpy.shutdown()
