class GlobalRegistry:
    def __init__(self):
        self._storage = {}
        self._subscribers = {}

    def register(self, key, value):
        self._storage[key] = value
        self._notify_subscribers(key, value)

    def get(self, key):
        return self._storage.get(key)

    def deregister(self, key):
        if key in self._storage:
            del self._storage[key]
            self._notify_subscribers(key, None)

    def keys(self):
        return self._storage.keys()

    def values(self):
        return self._storage.values()

    def items(self):
        return self._storage.items()

    def clear(self):
        self._storage.clear()

    def subscribe(self, key, callback):
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append(callback)

    def unsubscribe(self, key, callback):
        if key in self._subscribers:
            self._subscribers[key].remove(callback)
            if not self._subscribers[key]:
                del self._subscribers[key]

    def _notify_subscribers(self, key, value):
        if key in self._subscribers:
            for callback in self._subscribers[key]:
                callback(key, value)


registry = GlobalRegistry()

__all__ = ['registry']
