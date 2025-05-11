import msgpack
import zmq

class RemoteAPIClient:
    def __init__(self, host='localhost', port=23000):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        self._id = 0

    def getObject(self, object_name):
        return RemoteAPIObject(self, object_name)

    def call(self, object_name, function, args=[]):
        self._id += 1
        request = [0, self._id, object_name, function, args]
        self.socket.send(msgpack.packb(request, use_bin_type=True))
        response = msgpack.unpackb(self.socket.recv(), raw=False)
        if response[0] != 1:
            raise RuntimeError(f"Remote API call failed: {response}")
        return response[4] if len(response) > 4 else None

class RemoteAPIObject:
    def __init__(self, client, name):
        self.client = client
        self.name = name

    def __getattr__(self, function):
        def wrapper(*args):
            return self.client.call(self.name, function, list(args))
        return wrapper
