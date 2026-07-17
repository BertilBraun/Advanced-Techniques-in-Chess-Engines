import hashlib
import zipfile
from pathlib import Path

from tools.prepare_opening_suite import (
    ARCHIVE_MEMBER,
    prepare_opening_suite,
)
from src.experiment.evaluation_protocol import load_opening_suite


PGN = """[Event "First"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *

[Event "Second"]
[Result "*"]

1. d4 d5 2. c4 e6 *
"""


def test_prepare_opening_suite_is_deterministic(tmp_path: Path) -> None:
    archive_path = tmp_path / 'book.zip'
    output_path = tmp_path / 'openings.tsv'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr(ARCHIVE_MEMBER, PGN)
    expected_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    prepare_opening_suite(
        archive_path=archive_path,
        output_path=output_path,
        opening_count=2,
        random_seed=7,
        expected_sha256=expected_sha256,
    )

    first_output = output_path.read_text(encoding='utf-8')
    prepare_opening_suite(
        archive_path=archive_path,
        output_path=output_path,
        opening_count=2,
        random_seed=7,
        expected_sha256=expected_sha256,
    )

    assert output_path.read_text(encoding='utf-8') == first_output
    assert len(load_opening_suite(output_path)) == 2
