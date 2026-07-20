import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_chess_settings_does_not_import_pygame() -> None:
    source = """
import sys
import src.games.chess.ChessSettings

assert 'pygame' not in sys.modules
assert 'src.eval.GridGUI' not in sys.modules
"""

    subprocess.run(
        [sys.executable, '-c', source],
        cwd=PROJECT_ROOT,
        check=True,
    )
