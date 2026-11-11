import os

from llm_engineering.settings import settings

# In offline mode, disable Opik tracing entirely so no cloud calls (and 401s) occur.
if settings.is_offline:
    os.environ.setdefault("OPIK_DISABLED", "true")
    try:
        import opik

        def _noop_track(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda f: f

        opik.track = _noop_track
    except Exception:
        pass

from llm_engineering import application, domain, infrastructure

__all__ = ["settings", "application", "domain", "infrastructure"]
