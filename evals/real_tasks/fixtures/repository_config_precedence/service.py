from config import load_config

FALLBACK_PORT = 7000


def startup_settings(env, flags):
    settings = load_config(env, flags)
    return {
        "port": FALLBACK_PORT if settings["port"] is None else settings["port"],
        "debug": bool(settings["debug"]),
    }
