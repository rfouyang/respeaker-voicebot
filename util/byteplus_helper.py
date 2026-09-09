"""Shared base for the BytePlus voice clients."""

from __future__ import annotations

import os

from config.settings import BytePlusConfig, use_utf8_output


class BytePlusHelper:
    """Load and validate the BytePlus API key once for every voice client."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv(BytePlusConfig.API_KEY_ENV)
        if not self.api_key:
            raise ValueError(f"{BytePlusConfig.API_KEY_ENV} is missing from .env")


def demo_key_loads() -> None:
    """The key is present and long enough to be a real one."""
    helper = BytePlusHelper()
    print(f"API key 已加载，长度 {len(helper.api_key)}，前缀 {helper.api_key[:4]}...")
    assert len(helper.api_key) > 8


def main() -> None:
    use_utf8_output()
    demo_key_loads()


if __name__ == "__main__":
    main()
