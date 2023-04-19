import functools
import threading
import traceback
from multiprocessing import Queue
from threading import Thread


def synchronized(function):
    lock = threading.Lock()

    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        with lock:
            return function(self, *args, **kwargs)

    return wrapper


class ResponseStream:
    IS_STOPPED = False

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now or ResponseStream.IS_STOPPED:
                self.stop_now = False
                ResponseStream.IS_STOPPED = False
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        self.q.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        self.q.close()

    @synchronized
    def stop(self):
        ResponseStream.IS_STOPPED = True
