from __future__ import annotations

from pathlib import Path

import prepare_evaluation_artifacts
import pytest


def test_python_nvidia_libraries_are_prepended_for_engine_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = tmp_path / 'site-packages'
    cublas = site_packages / 'nvidia' / 'cublas' / 'lib'
    cudnn = site_packages / 'nvidia' / 'cudnn' / 'lib'
    cublas.mkdir(parents=True)
    cudnn.mkdir(parents=True)
    monkeypatch.setattr(prepare_evaluation_artifacts.site, 'getsitepackages', lambda: [str(site_packages)])
    monkeypatch.setenv('LD_LIBRARY_PATH', '/existing')

    prepare_evaluation_artifacts.configure_python_nvidia_libraries()

    assert prepare_evaluation_artifacts.os.environ['LD_LIBRARY_PATH'] == (
        f'{cublas}{prepare_evaluation_artifacts.os.pathsep}{cudnn}{prepare_evaluation_artifacts.os.pathsep}/existing'
    )
