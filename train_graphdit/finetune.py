"""Convenience entry point for conditional RPPD fine-tuning."""

from __future__ import annotations

import sys

from train import main


if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        main()
    if "--pretrained-model" not in sys.argv:
        raise SystemExit("finetune.py requires --pretrained-model PATH")
    if not any(option == "--property" or option.startswith("--property=") or option == "--all"
               for option in sys.argv):
        raise SystemExit("finetune.py requires --property NAME or --all")
    main()
