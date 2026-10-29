def lookup_setting(settings: dict, key: str, default=None):
    return settings.get(key, default)


def overlay_settings(defaults: dict, overrides: dict) -> dict:
    result = defaults.copy()
    result.update(overrides)
    return result
