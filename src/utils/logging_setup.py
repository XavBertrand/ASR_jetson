"""
Configuration logging avec loguru
"""
import sys
from pathlib import Path
from loguru import logger
from functools import wraps
import time
from contextlib import contextmanager


def setup_logging(
        log_dir: Path = Path("logs"),
        level: str = "INFO",
        rotation: str = "500 MB",
        retention: str = "10 days",
        json_logs: bool = False,
) -> None:
    """
    Configure le logging avec loguru

    Args:
        log_dir: Dossier des logs
        level: Niveau (DEBUG, INFO, WARNING, ERROR)
        rotation: Rotation des fichiers
        retention: Durée de rétention
        json_logs: Format JSON pour parsing
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Supprimer handler par défaut
    logger.remove()

    # Console (coloré)
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Fichier (tous les logs)
    logger.add(
        log_dir / "asr_{time:YYYY-MM-DD}.log",
        rotation=rotation,
        retention=retention,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        enqueue=True,
    )

    # Fichier erreurs
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        rotation=rotation,
        retention=retention,
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
        backtrace=True,
        diagnose=True,
    )

    # JSON (optionnel)
    if json_logs:
        logger.add(
            log_dir / "asr_{time:YYYY-MM-DD}.json",
            rotation=rotation,
            retention=retention,
            level=level,
            serialize=True,
        )

    logger.info(f"Logging initialized: level={level}, dir={log_dir}")


def log_system_info():
    """Log infos système"""
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


def log_performance(func):
    """Décorateur pour mesurer performance"""

    @wraps(func)
    def wrapper(*args, **kwargs):
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
def log_step(step_name: str):
    """Context manager pour logger une étape"""
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