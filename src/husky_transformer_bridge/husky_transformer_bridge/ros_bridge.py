"""Isolated phase-1 RPC bridge; no motion publishers or navigation clients."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import signal
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from husky_transformer_policy.protocol import request

class Bridge(Node):
    def __init__(self,path):
        super().__init__('policy_bridge',namespace='/transformer/shadow')
        self.path=path
        self.pool=ThreadPoolExecutor(max_workers=1)
        self.future=None
        self.ident=None
        self.pub=self.create_publisher(String,'response',10)
        self.sub=self.create_subscription(String,'request',self.receive,10)
        self.timer=self.create_timer(.1,self.poll)

    def emit(self,ident,**payload):
        self.pub.publish(String(data=json.dumps(dict(client_id=ident,**payload),allow_nan=False)))

    def receive(self,msg):
        ident=None
        try:
            if len(msg.data.encode())>65536: raise ValueError('Message too large')
            q=json.loads(msg.data)
            ident=q.pop('client_id')
            if not isinstance(ident,str) or not ident: raise ValueError('Missing client_id')
            if self.future is not None:
                self.emit(ident,ok=False,error='BUSY: retry explicitly')
                return
            if q.get('op') not in ('init_task','status','infer','commit','reset'):
                raise ValueError('Unknown operation')
            self.ident=ident
            self.future=self.pool.submit(request,self.path,timeout=1.0,**q)
        except Exception as exc: self.emit(ident,ok=False,error=str(exc))

    def poll(self):
        if self.future is None or not self.future.done(): return
        try: self.emit(self.ident,ok=True,result=self.future.result())
        except Exception as exc: self.emit(self.ident,ok=False,error=str(exc))
        finally: self.future=None;self.ident=None

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.destroy_node()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--socket',required=True)
    p.add_argument('--domain-id',type=int,default=42)
    a=p.parse_args()
    if not 1 <= a.domain_id <= 232: raise ValueError('Use an isolated domain in 1..232, never robot domain 0')
    rclpy.init(domain_id=a.domain_id)
    signal.signal(signal.SIGINT,signal.default_int_handler)
    node=Bridge(a.socket)
    try: rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException): pass
    finally:
        node.close()
        if rclpy.ok(): rclpy.shutdown()

if __name__=='__main__': main()
