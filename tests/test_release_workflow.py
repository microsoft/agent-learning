"""Release workflow contract tests."""

from pathlib import Path


def test_release_job_uses_flattened_linux_artifact_path() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish.yaml"
    ).read_text(encoding="utf-8")

    expected = 'release-assets/agent-learning-cli-${VERSION}-linux-x64.tar.gz'
    obsolete = (
        'release-assets/dist-installer/'
        'agent-learning-cli-${VERSION}-linux-x64.tar.gz'
    )
    assert expected in workflow
    assert obsolete not in workflow


def test_release_job_uses_flattened_windows_artifact_paths() -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish.yaml"
    ).read_text(encoding="utf-8")

    for suffix in ("exe", "zip"):
        filename = f"agent-learning-cli-${{VERSION}}-windows-x64.{suffix}"
        assert f'release-assets/{filename}' in workflow
        assert f'release-assets/dist-installer/{filename}' not in workflow


def test_windows_installer_uses_directory_bundle_and_precedes_old_shims() -> None:
    root = Path(__file__).resolve().parents[1]
    build_script = (root / "scripts" / "build_windows_installer.ps1").read_text(
        encoding="utf-8"
    )
    installer = (
        root / "packaging" / "windows" / "agent-learning-installer.iss"
    ).read_text(encoding="utf-8")

    assert "--onedir" in build_script
    assert "--onefile" not in build_script
    assert "$env:LOCALAPPDATA" in build_script
    assert 'Source: "dist\\agent-learn\\*"' in installer
    assert "recursesubdirs" in installer
    assert "PrivilegesRequired=lowest" in installer
    assert 'ValueData: "{code:PathWithEntryFirst|{app}}"' in installer
    assert 'ValueData: "{olddata};{app}"' not in installer