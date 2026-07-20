import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_training_entrypoint_does_not_import_pygame() -> None:
    source = """
import sys
import src.train.Trainer

assert 'pygame' not in sys.modules
assert 'src.eval.GridGUI' not in sys.modules
"""

    subprocess.run(
        [sys.executable, '-c', source],
        cwd=PROJECT_ROOT,
        check=True,
    )
