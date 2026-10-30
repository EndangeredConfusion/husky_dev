"""No GPU/ROS required; timeout and disconnection behavior."""
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
from husky_transformer_policy.protocol import request

with tempfile.TemporaryDirectory() as d:
    path=str(Path(d)/'test.sock')
    try:request(path,'status',timeout=.1)
    except OSError:pass
    else:raise AssertionError('Missing worker accepted')
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(path);server.listen(1)
        def hang():
            conn,_=server.accept()
            with conn:time.sleep(.4)
        t=threading.Thread(target=hang);t.start()
        start=time.monotonic()
        try:request(path,'status',timeout=.1)
        except TimeoutError:pass
        else:raise AssertionError('Unresponsive worker accepted')
        elapsed=time.monotonic()-start
        assert elapsed<.3
        t.join()
result=dict(status='PASS',missing_worker_rejected=True,timeout_seconds=elapsed)
Path('transport_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
