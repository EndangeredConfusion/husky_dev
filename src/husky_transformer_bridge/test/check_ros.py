"""System Python test client; requires isolated bridge in domain 42."""
import argparse
import json
from pathlib import Path
import time
import rclpy
from std_msgs.msg import String

p=argparse.ArgumentParser();p.add_argument('--expect-unavailable',action='store_true');a=p.parse_args()
rclpy.init(domain_id=42)
n=rclpy.create_node('phase1_test_client')
pub=n.create_publisher(String,'/transformer/shadow/request',10)
answers={}
def receive(msg):
    value=json.loads(msg.data);answers[value['client_id']]=value
sub=n.create_subscription(String,'/transformer/shadow/response',receive,10)
counter=0
def call(op,**kw):
    global counter
    counter+=1;ident=str(counter)
    pub.publish(String(data=json.dumps(dict(client_id=ident,op=op,**kw))))
    end=time.monotonic()+3
    while ident not in answers and time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
    assert ident in answers,'ROS response timeout'
    return answers[ident]
try:
    end=time.monotonic()+8
    while pub.get_subscription_count()==0 and time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.1)
    assert pub.get_subscription_count()==1
    if a.expect_unavailable:
        result=call('status');assert not result['ok'] and 'result' not in result
        output='ros_unavailable_result.json'
    else:
        result=call('init_task',lambda_values=[7,2,2,2]);assert result['ok']
        epoch=result['result']['epoch']
        for i in range(10):
            result=call('infer',epoch=epoch,history_version=0,agent_pos=[4,3],state_seq=i)
            assert result['ok'] and result['result']['goal_index']==0
            assert result['result']['history_size']==0 and result['result']['state_seq']==i
        bad=call('infer',epoch='stale',history_version=0,agent_pos=[4,3]);assert not bad['ok']
        output='ros_roundtrip_result.json'
    summary=dict(status='PASS',requests=counter,last_response=result)
    Path(output).write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
finally:n.destroy_node();rclpy.shutdown()
