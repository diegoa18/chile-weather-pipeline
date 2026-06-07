import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str | Path | None = None) -> None:
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    root.addHandler(handler)

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
