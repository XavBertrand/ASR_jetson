"""
Logging configuration utilities built on top of loguru.
"""
import sys
from pathlib import Path
from loguru import logger
from functools import wraps
import time
from contextlib import contextmanager
from typing import Callable, Iterator, ParamSpec, TypeVar


P = ParamSpec("P")
T = TypeVar("T")


def setup_logging(
        log_dir: Path = Path("logs"),
        level: str = "INFO",
        rotation: str = "500 MB",
        retention: str = "10 days",
        json_logs: bool = False,
) -> None:
    """
    Configure the global loguru logger.

    :param log_dir: Directory where log files are stored.
    :type log_dir: Path
    :param level: Verbosity level (`DEBUG`, `INFO`, `WARNING`, or `ERROR`).
    :type level: str
    :param rotation: Log rotation policy understood by loguru.
    :type rotation: str
    :param retention: Retention policy for rotated log files.
    :type retention: str
    :param json_logs: Set to ``True`` to add a JSON log sink.
    :type json_logs: bool
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove the default handler.
    logger.remove()

    # Colorized console output.
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Full debug log file.
    logger.add(
        log_dir / "asr_{time:YYYY-MM-DD}.log",
        rotation=rotation,
        retention=retention,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        enqueue=True,
    )

    # Error-only log file.
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        rotation=rotation,
        retention=retention,
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
        backtrace=True,
        diagnose=True,
    )

    # Optional JSON sink for machine parsing.
    if json_logs:
        logger.add(
            log_dir / "asr_{time:YYYY-MM-DD}.json",
            rotation=rotation,
            retention=retention,
            level=level,
            serialize=True,
        )

    logger.info(f"Logging initialized: level={level}, dir={log_dir}")


def log_system_info() -> None:
    """
    Log high-level system information such as platform, Python, and GPU data.
    """
    import platform
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")

    if has_torch:
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU memory: {mem:.2f} GB")

    logger.info("=" * 60)


def log_performance(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator that records execution duration and outcome for a function.

    :param func: Function whose execution should be instrumented.
    :type func: Callable
    :returns: Wrapped function with logging side effects.
    :rtype: Callable
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        logger.debug(f"Starting {func.__name__}")

        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            logger.info(f"✓ {func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.perf_counter() - start
            logger.error(f"✗ {func.__name__} failed after {duration:.3f}s: {e}")
            raise

    return wrapper


@contextmanager
def log_step(step_name: str) -> Iterator[None]:
    """
    Context manager that logs the start, success, or failure of a processing step.

    :param step_name: Human-friendly name of the step being executed.
    :type step_name: str
    :yields: ``None`` – execution continues inside the managed block.
    :rtype: Iterator[None]
    """
    start = time.perf_counter()
    logger.info(f"→ Starting: {step_name}")

    try:
        yield
        duration = time.perf_counter() - start
        logger.success(f"✓ Completed: {step_name} ({duration:.2f}s)")
    except Exception as e:
        duration = time.perf_counter() - start
        logger.error(f"✗ Failed: {step_name} after {duration:.2f}s")
        logger.exception(e)
        raise
