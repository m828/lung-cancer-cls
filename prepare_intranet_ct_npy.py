#!/usr/bin/env python3
"""Entry point for intranet CT DICOM QC and NPY conversion."""

from __future__ import annotations

import sys
from pathlib import Path


src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from lung_cancer_cls.intranet_ct_preprocess import main  # noqa: E402


if __name__ == "__main__":
    main()
