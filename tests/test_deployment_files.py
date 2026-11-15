"""The deployment files: ``compose.yaml``, ``Dockerfile`` and
``docker/init-store.sh``, checked without Docker."""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Final

import dashboard_seed as seed
import pytest
import yaml

from keepout._environment import parse_time

REPO: Final = Path(__file__).parents[1]
INIT: Final = REPO / "docker" / "init-store.sh"


def _compose() -> dict[str, Any]:
    data = yaml.safe_load((REPO / "compose.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _after(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_compose_file() -> None:
    services = _compose()["services"]
    dashboard, worker = services["dashboard"], services["worker"]
    assert dashboard["ports"] == ["127.0.0.1:${KEEPOUT_PORT:-8501}:8501"]
    assert _after(dashboard["command"], "--host") == "0.0.0.0"  # noqa: S104 (inside the container)
    as_of = parse_time(_after(dashboard["command"], "--as-of"))
    assert as_of.utc.isot == seed.DEMO_AS_OF.utc.isot
    for service in (dashboard, worker):
        assert service["depends_on"] == {
            "init": {"condition": "service_completed_successfully"}
        }
        assert service["volumes"] == ["data:/data"]
        # Both run child processes (a job, Streamlit): an init reaps them.
        assert service["init"] is True
    assert worker["stop_signal"] == "SIGTERM"
    assert worker["stop_grace_period"] == "30s"
    assert worker["restart"] == "unless-stopped"
    assert "ports" not in worker
    assert "ports" not in services["init"]


def test_dockerfile_pins_and_user() -> None:
    lines = (REPO / "Dockerfile").read_text(encoding="utf-8").splitlines()
    from_lines = [line for line in lines if line.startswith("FROM ")]
    assert from_lines
    external = [line for line in from_lines if "@" in line or ":" in line.split()[1]]
    assert all(re.search(r"@sha256:[0-9a-f]{64}\b", line) for line in external)
    last = max(i for i, line in enumerate(lines) if line.startswith("FROM "))
    stage = lines[last:]
    user = [i for i, line in enumerate(stage) if line.startswith("USER ")]
    copies = [i for i, line in enumerate(stage) if line.startswith("COPY ")]
    assert [stage[i] for i in user] == ["USER keepout"]
    assert user[0] > max(copies)
    syncs = " ".join(line for line in lines if "uv sync" in line)
    assert "--locked --no-dev" in syncs


def _init(seed_dir: Path, data_dir: Path) -> subprocess.CompletedProcess[str]:
    sh = shutil.which("sh")
    assert sh is not None
    return subprocess.run(  # noqa: S603 (a fixed script)
        [sh, str(INIT)],
        env={
            "PATH": "/usr/bin:/bin",
            "KEEPOUT_SEED_DIR": str(seed_dir),
            "KEEPOUT_DATA_DIR": str(data_dir),
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def seed_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "seed"
    (directory / "keepout.db.catalogs").mkdir(parents=True)
    (directory / "keepout.db").write_bytes(b"seed store")
    (directory / "keepout.db.catalogs" / "abc.csv.gz").write_bytes(b"catalog")
    return directory


def test_init_copies_then_keeps(seed_dir: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    first = _init(seed_dir, data)
    assert first.returncode == 0, first.stderr
    assert first.stdout == f"demo store copied to {data / 'keepout.db'}\n"
    assert (data / "keepout.db").read_bytes() == b"seed store"
    assert (data / "keepout.db.catalogs" / "abc.csv.gz").read_bytes() == b"catalog"
    assert sorted(p.name for p in data.iterdir()) == [
        "keepout.db",
        "keepout.db.catalogs",
    ]


def test_init_keeps_existing_store(seed_dir: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "keepout.db").write_bytes(b"my store")
    result = _init(seed_dir, data)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"store kept: {data / 'keepout.db'}\n"
    assert (data / "keepout.db").read_bytes() == b"my store"
    assert not (data / "keepout.db.catalogs").exists()
