from settings import DEFAULT_TTL


def read_value(cache, key, fetch, *, now, ttl=DEFAULT_TTL):
    status, value = cache.get(key, now=now)
    if status == "hit":
        return value
    value = fetch(key)
    cache.put(key, value, now=now, ttl=ttl)
    return value
