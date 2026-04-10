#!/usr/bin/env python3
"""Create a manual-review flags template from an intranet CT manifest."""

from __future__ import annotations

import sys
from pathlib import Path


src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from lung_cancer_cls.intranet_ct_review import main_generate_review_flags  # noqa: E402


if __name__ == "__main__":
    main_generate_review_flags()
