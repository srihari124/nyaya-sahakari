import os
import logging

logger = logging.getLogger(__name__)

if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
    os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL")

LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_SECRET_KEY")) and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))

if LANGFUSE_ENABLED:
    try:
        from langfuse.decorators import observe as _observe, langfuse_context as _langfuse_context
        from langfuse import Langfuse as _Langfuse
        _client = _Langfuse()
        observe = _observe
        langfuse_context = _langfuse_context
        logger.info(
            "Langfuse tracing enabled: host=%s",
            os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception as exc:
        LANGFUSE_ENABLED = False
        logger.warning("Langfuse init failed, tracing disabled: %s", exc)

if not LANGFUSE_ENABLED:
    def observe(func=None, **kwargs):
        """No-op @observe decorator when Langfuse is disabled."""
        if func is not None:
            return func
        def decorator(fn):
            return fn
        return decorator

    class langfuse_context:  # noqa: N801
        """No-op context when Langfuse is disabled."""
        @staticmethod
        def update_current_observation(**kwargs): pass
        @staticmethod
        def update_current_trace(**kwargs): pass
        @staticmethod
        def score_current_trace(**kwargs): pass
