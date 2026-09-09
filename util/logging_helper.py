"""Shared Loguru setup for right-click voice entry points."""

from __future__ import annotations

from pathlib import Path
import sys

from loguru import logger

from config.settings import use_utf8_output


def configure_logging(debug_dir: Path | None = None) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | {message}"
        ),
    )
    if debug_dir is not None:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        logger.add(
            Path(debug_dir) / "runtime.log",
            level="DEBUG",
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        )


def demo_console_only() -> None:
    configure_logging()
    logger.info("控制台日志")
    logger.success("成功级别")
    logger.warning("警告级别")


def demo_writes_file() -> None:
    from config.settings import OutputConfig

    debug_dir = OutputConfig.OUTPUT_DIR / "logging_demo"
    configure_logging(debug_dir)
    logger.info("这一行同时写进文件")
    path = debug_dir / "runtime.log"
    assert path.exists() and "这一行" in path.read_text(encoding="utf-8")
    print(f"日志文件已写入: {path}")


def main() -> None:
    use_utf8_output()
    demo_console_only()
    demo_writes_file()


if __name__ == "__main__":
    main()
