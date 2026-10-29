_MISSING = object()


class Cache:
    def __init__(self):
        self._items = {}

    def put(self, key, value, *, now, ttl):
        self._items[key] = (value, now + ttl)

    def get(self, key, *, now):
        item = self._items.get(key, _MISSING)
        if item is _MISSING:
            return "miss", None
        value, expires = item
        if now >= expires:
            del self._items[key]
            return "expired", None
        return "hit", value
