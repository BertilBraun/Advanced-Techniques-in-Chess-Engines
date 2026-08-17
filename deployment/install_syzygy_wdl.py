from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import quote
from urllib.request import urlopen

import chess
import chess.syzygy


DEFAULT_DESTINATION = Path('/workspace/syzygy/wdl345')
DEFAULT_SOURCE_BASE_URL = 'https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl'
FILE_NAME_SOURCE_URL = (
    'https://raw.githubusercontent.com/syzygy1/tb/576194c56faf671328b12529a4d7454cb6e4fabb/checksums/wdl345.txt'
)
DOWNLOAD_CHECKSUM_SOURCE_URL = 'https://tablebase.lichess.ovh/tables/standard/sha512'
DOWNLOAD_MANIFEST_PATH = Path(__file__).with_name('syzygy-wdl345.sha512')
EXPECTED_FILE_COUNT = 145
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

_DOWNLOAD_CHECKSUM_LINE = re.compile(r'^([0-9a-f]{128})  ([A-Za-z0-9]+\.rtbw)$')


@dataclass(frozen=True)
class WdlFile:
    name: str
    sha512: str


def parse_download_manifest(checksum_manifest: str) -> tuple[WdlFile, ...]:
    files = tuple(
        WdlFile(name=match.group(2), sha512=match.group(1))
        for line in checksum_manifest.splitlines()
        if (match := _DOWNLOAD_CHECKSUM_LINE.fullmatch(line)) is not None
    )
    if len(files) != EXPECTED_FILE_COUNT or len({file.name for file in files}) != EXPECTED_FILE_COUNT:
        raise ValueError(f'Expected {EXPECTED_FILE_COUNT} unique 3-5-piece WDL checksums.')
    return files


def install_wdl_files(destination: Path, source_base_url: str, files: tuple[WdlFile, ...]) -> None:
    if destination.exists():
        _verify_installation(destination, files)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.syzygy-wdl345-', dir=destination.parent) as temporary_directory:
        temporary_path = Path(temporary_directory)
        for index, file in enumerate(files, start=1):
            print(f'Downloading Syzygy WDL file {index}/{len(files)}: {file.name}', flush=True)
            source_url = f'{source_base_url.rstrip("/")}/{quote(file.name)}'
            _download_verified_file(source_url, temporary_path / file.name, file.sha512)
        _write_installation_record(temporary_path, source_base_url, files)
        temporary_path.replace(destination)
    _verify_installation(destination, files)


def _download_verified_file(url: str, destination: Path, expected_sha512: str) -> None:
    last_error: OSError | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        digest = hashlib.sha512()
        try:
            with urlopen(url, timeout=60) as response, destination.open('wb') as output:
                while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                    output.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha512:
                destination.unlink()
                raise ValueError(f'Download checksum mismatch for {url}.')
            return
        except OSError as error:
            last_error = error
            destination.unlink(missing_ok=True)
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(attempt)
    assert last_error is not None
    raise last_error


def _verify_installation(destination: Path, files: tuple[WdlFile, ...]) -> None:
    if not destination.is_dir():
        raise ValueError(f'Syzygy WDL destination is not a directory: {destination}')
    expected_names = {file.name for file in files}
    actual_names = {path.name for path in destination.glob('*.rtbw')}
    if actual_names != expected_names:
        raise ValueError('Syzygy WDL installation does not contain the exact pinned 3-5-piece file set.')
    if any(destination.glob('*.rtbz')):
        raise ValueError('Syzygy WDL installation must not contain DTZ files.')
    for file in files:
        digest = hashlib.sha512()
        with (destination / file.name).open('rb') as input_file:
            while chunk := input_file.read(DOWNLOAD_CHUNK_BYTES):
                digest.update(chunk)
        if digest.hexdigest() != file.sha512:
            raise ValueError(f'Syzygy WDL checksum mismatch: {file.name}')


def _write_installation_record(destination: Path, source_base_url: str, files: tuple[WdlFile, ...]) -> None:
    (destination / 'INSTALLATION.txt').write_text(
        '\n'.join(
            (
                'Syzygy tables: 3-5-piece WDL only',
                f'Source: {source_base_url}',
                f'Files: {len(files)}',
                f'File-name source: {FILE_NAME_SOURCE_URL}',
                f'Download-checksum source: {DOWNLOAD_CHECKSUM_SOURCE_URL}',
                f'Pinned manifest SHA-256: {hashlib.sha256(DOWNLOAD_MANIFEST_PATH.read_bytes()).hexdigest()}',
                '',
            )
        ),
        encoding='utf-8',
    )


def _smoke_tablebase(destination: Path) -> None:
    winning_position = chess.Board('7k/8/8/8/8/8/6Q1/6K1 w - - 0 1')
    uncovered_position = chess.Board('7k/7p/8/8/8/8/4PQR1/6K1 w - - 0 1')
    with chess.syzygy.open_tablebase(str(destination), load_dtz=False) as tablebase:
        if tablebase.probe_wdl(winning_position) != 2:
            raise RuntimeError('Syzygy WDL smoke did not recognize a winning KQvK position.')
        try:
            tablebase.probe_wdl(uncovered_position)
        except KeyError:
            pass
        else:
            raise RuntimeError('The 3-5-piece Syzygy installation unexpectedly covers a six-piece position.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Install the pinned 3-5-piece Syzygy WDL tablebase set.')
    parser.add_argument('--destination', type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument('--source-base-url', default=DEFAULT_SOURCE_BASE_URL)
    arguments = parser.parse_args()

    files = parse_download_manifest(DOWNLOAD_MANIFEST_PATH.read_text(encoding='ascii'))
    install_wdl_files(arguments.destination, arguments.source_base_url, files)
    _smoke_tablebase(arguments.destination)
    print(f'Installed and verified {len(files)} Syzygy WDL files in {arguments.destination}.')


if __name__ == '__main__':
    main()
