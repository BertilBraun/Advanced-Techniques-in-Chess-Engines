from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from deployment.install_syzygy_wdl import EXPECTED_FILE_COUNT, WdlFile, install_wdl_files, parse_download_manifest


def _checksum_manifest(contents: tuple[bytes, ...]) -> str:
    return '\n'.join(
        f'{hashlib.sha512(content).hexdigest()}  K{index}vK.rtbw' for index, content in enumerate(contents)
    )


def test_download_manifest_requires_exact_3_to_5_piece_file_set() -> None:
    contents = tuple(str(index).encode('ascii') for index in range(EXPECTED_FILE_COUNT))

    files = parse_download_manifest(_checksum_manifest(contents))

    assert len(files) == EXPECTED_FILE_COUNT
    assert files[0] == WdlFile(name='K0vK.rtbw', sha512=hashlib.sha512(contents[0]).hexdigest())
    assert files[-1].name == f'K{EXPECTED_FILE_COUNT - 1}vK.rtbw'


def test_download_manifest_rejects_wrong_file_count() -> None:
    contents = tuple(str(index).encode('ascii') for index in range(EXPECTED_FILE_COUNT - 1))

    with pytest.raises(ValueError, match='145 unique'):
        parse_download_manifest(_checksum_manifest(contents))


def test_installer_downloads_verifies_and_reuses_existing_files(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()
    contents = (b'first table', b'second table')
    files = tuple(
        WdlFile(name=f'K{index}vK.rtbw', sha512=hashlib.sha512(content).hexdigest())
        for index, content in enumerate(contents)
    )
    for file, content in zip(files, contents, strict=True):
        (source / file.name).write_bytes(content)
    destination = tmp_path / 'installed'

    install_wdl_files(destination, source.as_uri(), files)
    assert (destination / 'INSTALLATION.txt').is_file()
    install_wdl_files(destination, source.as_uri(), files)

    (destination / files[0].name).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        install_wdl_files(destination, source.as_uri(), files)
