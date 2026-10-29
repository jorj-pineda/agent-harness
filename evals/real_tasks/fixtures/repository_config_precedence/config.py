from defaults import DEFAULTS


def load_config(env, flags):
    settings = DEFAULTS.copy()
    if "PORT" in env:
        settings["port"] = int(env["PORT"])
    if "DEBUG" in env:
        settings["debug"] = env["DEBUG"].strip().lower() in {"1", "true", "yes"}
    for key in ("port", "debug"):
        if key in flags:
            settings[key] = flags[key]
    return settings
