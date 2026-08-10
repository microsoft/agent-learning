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