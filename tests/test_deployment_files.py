"""The deployment files: ``compose.yaml``, ``Dockerfile`` and
``docker/init-store.sh``, checked without Docker."""

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Final

import dashboard_seed as seed
import iridium_cosmos
import pytest
import yaml
from builders import store_at

from keepout._environment import parse_time
from keepout.store import open_store

REPO: Final = Path(__file__).parents[1]
INIT: Final = REPO / "docker" / "init-store.sh"


def _compose() -> dict[str, Any]:
    data = yaml.safe_load((REPO / "compose.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _after(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_compose_publishes_on_loopback_at_the_demo_time() -> None:
    services = _compose()["services"]
    dashboard, worker = services["dashboard"], services["worker"]
    assert dashboard["ports"] == ["127.0.0.1:${KEEPOUT_PORT:-8501}:8501"]
    assert _after(dashboard["command"], "--host") == "0.0.0.0"  # noqa: S104 (inside the container)
    default = re.fullmatch(
        r"\$\{KEEPOUT_AS_OF-(.+)\}", dashboard["environment"]["KEEPOUT_AS_OF"]
    )
    assert default is not None
    as_of = parse_time(default[1])
    assert as_of.utc.isot == seed.DEMO_AS_OF.utc.isot
    assert "ports" not in worker
    assert "ports" not in services["init"]


def test_compose_iridium_profile() -> None:
    services = _compose()["services"]
    iridium = {name: service for name, service in services.items() if "iridium" in name}
    assert set(iridium) == {"iridium-init", "iridium-dashboard", "iridium-worker"}
    for service in iridium.values():
        assert service["profiles"] == ["iridium"]
        assert service["volumes"] == ["iridium-data:/data"]
    for name, service in services.items():
        if name not in iridium:
            assert "profiles" not in service
            assert all("iridium" not in volume for volume in service.get("volumes", []))
    dashboard = iridium["iridium-dashboard"]
    assert dashboard["ports"] == ["127.0.0.1:${KEEPOUT_IRIDIUM_PORT:-8502}:8501"]
    as_of = parse_time(dashboard["environment"]["KEEPOUT_AS_OF"])
    assert as_of.utc.isot == iridium_cosmos.IRIDIUM_AS_OF.utc.isot
    worker = iridium["iridium-worker"]
    assert worker["command"] == ["keepout", "worker", "--store", "/data/keepout.db"]
    assert "ports" not in worker
    init = iridium["iridium-init"]
    assert init["command"] == [
        "/opt/keepout/init-store.sh",
        "/opt/keepout/seed/iridium",
    ]
    assert "iridium-data" in _compose()["volumes"]


def test_the_worker_s_healthcheck_reads_its_heartbeat_file() -> None:
    worker = _compose()["services"]["worker"]
    alive = worker["environment"]["KEEPOUT_HEARTBEAT_FILE"]
    assert f"os.stat('{alive}')" in worker["healthcheck"]["test"][-1]


def test_dockerfile_pins_and_user() -> None:
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    lines = text.splitlines()
    # Every image pulled (ARG PYTHON_IMAGE=python..., COPY
    # --from=ghcr.io/...), not the stages named in the file.
    images = re.findall(
        r"^(?:FROM |COPY --from=|ARG \w+_IMAGE=)(\S+)", text, re.MULTILINE
    )
    external = [image for image in images if ":" in image]
    assert len(external) >= 2
    assert all(re.search(r"@sha256:[0-9a-f]{64}$", image) for image in external)
    # The build and the runtime run on one Python image.
    stages = re.findall(r"^FROM (\S+)", text, re.MULTILINE)
    assert stages.count("${PYTHON_IMAGE}") == 2
    last = max(i for i, line in enumerate(lines) if line.startswith("FROM "))
    stage = lines[last:]
    user = [i for i, line in enumerate(stage) if line.startswith("USER ")]
    assert [stage[i] for i in user] == ["USER 10001:10001"]


def _init(seed_dir: Path, data_dir: Path) -> subprocess.CompletedProcess[str]:
    sh = shutil.which("sh")
    assert sh is not None
    # keepout (this environment's) checks a kept store's schema.
    path = f"{Path(sys.executable).parent}:/usr/bin:/bin"
    return subprocess.run(  # noqa: S603 (a fixed script)
        [sh, str(INIT), str(seed_dir), str(data_dir)],
        env={"PATH": path},
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


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    directory.mkdir()
    return directory


def test_init_copies_the_demo_store(seed_dir: Path, data_dir: Path) -> None:
    # A copy an earlier run left unfinished is replaced.
    (data_dir / ".init-keepout.db").write_bytes(b"partial")
    result = _init(seed_dir, data_dir)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"demo store copied to {data_dir / 'keepout.db'}\n"
    assert (data_dir / "keepout.db").read_bytes() == b"seed store"
    catalog = data_dir / "keepout.db.catalogs" / "abc.csv.gz"
    assert catalog.read_bytes() == b"catalog"
    assert sorted(p.name for p in data_dir.iterdir()) == [
        "keepout.db",
        "keepout.db.catalogs",
    ]


def test_init_keeps_a_current_store(seed_dir: Path, data_dir: Path) -> None:
    store = data_dir / "keepout.db"
    open_store(store).close()
    result = _init(seed_dir, data_dir)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"store kept: {store}\n"
    assert not (data_dir / "keepout.db.catalogs").exists()  # Not the demo's.


def test_init_fails_on_an_older_store(seed_dir: Path, data_dir: Path) -> None:
    store = data_dir / "keepout.db"
    store_at(store, "0002")
    before = store.read_bytes()
    result = _init(seed_dir, data_dir)
    assert result.returncode == 1
    assert "keepout store upgrade" in result.stderr
    assert store.read_bytes() == before


def test_init_fails_on_a_file_that_is_no_store_without_the_upgrade_hint(
    seed_dir: Path, data_dir: Path
) -> None:
    store = data_dir / "keepout.db"
    store.write_bytes(b"not a database, nor a store")
    result = _init(seed_dir, data_dir)
    assert result.returncode == 1
    assert f"{store} is not a keepout store" in result.stderr
    assert "upgrade" not in result.stderr
    assert store.read_bytes() == b"not a database, nor a store"


def test_init_keeps_catalogs_without_their_store(
    seed_dir: Path, data_dir: Path
) -> None:
    catalogs = data_dir / "keepout.db.catalogs"
    catalogs.mkdir()
    (catalogs / "mine.csv.gz").write_bytes(b"restored")
    result = _init(seed_dir, data_dir)
    assert result.returncode == 1
    assert "refusing" in result.stderr
    assert not (data_dir / "keepout.db").exists()
    assert (catalogs / "mine.csv.gz").read_bytes() == b"restored"


def test_python_classifiers_are_the_ci_matrix() -> None:
    """docs/stability.md: the versions tested in CI are the classifiers'."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    prefix = "Programming Language :: Python :: 3."
    classified = [
        "3." + item.removeprefix(prefix)
        for item in project["project"]["classifiers"]
        if item.startswith(prefix)
    ]
    ci = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text("utf-8"))
    assert ci["jobs"]["test"]["strategy"]["matrix"]["python-version"] == classified
