import logging

from app.gesture_engine.config import DEBUG_MODE, LOG_LEVEL

# ustala poziom logowania na podstawie LOG_LEVEL (np. "DEBUG", "INFO"); w razie bledu wraca do DEBUG_MODE
_level = getattr(logging, str(LOG_LEVEL).upper(), None)
if not isinstance(_level, int):
    _level = logging.DEBUG if DEBUG_MODE else logging.INFO

logging.basicConfig(
    level=_level,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

logger = logging.getLogger("inzynierka")
