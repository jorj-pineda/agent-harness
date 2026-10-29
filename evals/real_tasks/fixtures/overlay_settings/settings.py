def lookup_setting(settings: dict, key: str, default=None):
    return settings.get(key, default)
