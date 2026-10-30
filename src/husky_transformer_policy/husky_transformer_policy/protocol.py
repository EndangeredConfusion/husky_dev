"""Local, bounded JSON request/response transport. No ROS or torch dependency."""
import json
import socket
import uuid

LIMIT = 65536
VERSION = 1

def receive(sock):
    data = bytearray()
    while b'\n' not in data:
        chunk = sock.recv(min(4096, LIMIT + 1 - len(data)))
        if not chunk:
            raise ConnectionError('Disconnected before complete message')
        data.extend(chunk)
        if len(data) > LIMIT:
            raise ValueError('Message exceeds limit')
    line, trailing = data.split(b'\n', 1)
    if trailing:
        raise ValueError('Only one message is allowed per connection')
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError('Message must be an object')
    return value

def send(sock, value):
    data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()+b'\n'
    if len(data) > LIMIT:
        raise ValueError('Message exceeds limit')
    sock.sendall(data)

def request(path, op, timeout=1.0, **payload):
    ident = str(uuid.uuid4())
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(timeout)
        sock.connect(str(path))
        send(sock, dict(version=VERSION, request_id=ident, op=op, **payload))
        reply = receive(sock)
    if reply.get('version') != VERSION or reply.get('request_id') != ident:
        raise ValueError('Response identity mismatch')
    if not reply.get('ok'):
        raise RuntimeError(reply.get('error', 'Worker error'))
    return reply['result']
