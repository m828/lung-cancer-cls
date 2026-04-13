#!/usr/bin/env python3
"""Merge old intranet CT metadata with a newly rebuilt manifest."""

from __future__ import annotations

import sys
from pathlib import Path


src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from lung_cancer_cls.intranet_ct_manifest_merge import main  # noqa: E402


if __name__ == "__main__":
    main()
