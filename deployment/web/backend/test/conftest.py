from __future__ import annotations

import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[4]
sys.path.append(str(repository_root))
sys.path.append(str(repository_root / 'py'))
