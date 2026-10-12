#!/usr/bin/env python
"""Download RF-100VL (100 COCO-format detection datasets) into data/benchmarks/rf100vl/.

Needs a (free) Roboflow API key and the ``rf100vl`` package. The full set is
large (~41 GB, mostly train splits); evaluation only reads each dataset's
``test/`` split.

    pip install rf100vl
    ROBOFLOW_API_KEY=... python scripts/download_rf100vl.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.getenv("LLMSEG_DATA", REPO / "data" / "benchmarks")) / "rf100vl"


def main():
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY (free at https://roboflow.com)")
    import rf100vl

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"downloading RF-100VL -> {OUT} (COCO format, resumable)", flush=True)
    rf100vl.download_rf100vl(path=str(OUT), model_format="coco", overwrite=False, api_key=key)
    print("RF-100VL ready")


if __name__ == "__main__":
    main()
