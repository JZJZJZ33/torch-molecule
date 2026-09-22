"""Convenience entry point for unconditional Graph-DiT pretraining."""

from __future__ import annotations

import sys

from train import main


if __name__ == "__main__":
    if not any(option in sys.argv for option in ("--unconditional", "--property", "--all", "--list-properties")):
        sys.argv.insert(1, "--unconditional")
    main()
