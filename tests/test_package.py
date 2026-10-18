import importlib
import pkgutil
import subprocess
import sys
import urllib.request

import pytest

import keepout
from keepout import errors


def test_version_matches_distribution_metadata() -> None:
    from importlib.metadata import version

    assert keepout.__version__ == version("keepout")


def test_core_api_is_importable_from_the_package() -> None:
    from keepout.conjunction import Conjunction
    from keepout.pc import compute_pc
    from keepout.risk import ConjunctionEvent, assess
    from keepout.screening import (
        CloseApproach,
        ScreeningResult,
        ScreeningVolume,
        screen,
    )
    from keepout.state import StateVector

    assert keepout.Conjunction is Conjunction
    assert keepout.compute_pc is compute_pc
    assert keepout.StateVector is StateVector
    assert keepout.ConjunctionEvent is ConjunctionEvent
    assert keepout.assess is assess
    assert keepout.screen is screen
    assert keepout.CloseApproach is CloseApproach
    assert keepout.ScreeningResult is ScreeningResult
    assert keepout.ScreeningVolume is ScreeningVolume
    assert [name for name in keepout.__all__ if not hasattr(keepout, name)] == []


def test_every_error_and_method_type_is_exported_from_the_package() -> None:
    from keepout.pc import Method, MethodChoice

    defined = {
        name
        for name, value in vars(errors).items()
        if isinstance(value, type) and issubclass(value, errors.KeepoutError)
    }
    assert defined <= set(keepout.__all__)
    assert keepout.Method is Method
    assert keepout.MethodChoice is MethodChoice


PUBLIC_MODULES = sorted(
    module.name
    for module in pkgutil.walk_packages(keepout.__path__, "keepout.")
    if not any(part.startswith("_") for part in module.name.split("."))
)


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_public_modules_declare_resolvable_names(name: str) -> None:
    module = importlib.import_module(name)
    exported = module.__all__
    assert sorted(set(exported)) == sorted(exported)
    assert [item for item in exported if not hasattr(module, item)] == []


def test_tests_cannot_reach_the_network() -> None:
    with pytest.raises(AssertionError, match="must not access the network"):
        urllib.request.urlopen("https://example.com")


@pytest.mark.parametrize("module", ["keepout", "keepout.cli.app"])
def test_importing_loads_no_numerical_library(module: str) -> None:
    # keepout --help and import keepout stay fast: numpy, scipy and astropy
    # load on first use of a name that needs them.
    code = (
        f"import sys, {module}; "
        "print(sorted(m for m in ('numpy', 'scipy', 'astropy', 'erfa', 'sgp4') "
        "if m in sys.modules))"
    )
    loaded = subprocess.run(  # noqa: S603 (this interpreter, fixed code)
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert loaded.stdout == "[]\n"


def test_importing_reads_no_metadata() -> None:
    # importlib.metadata is most of the import time: __version__ is lazy.
    code = (
        "import sys, keepout; print('importlib.metadata' in sys.modules); "
        "print(keepout.__version__ == __import__('importlib.metadata').metadata"
        ".version('keepout'), '__version__' in dir(keepout))"
    )
    loaded = subprocess.run(  # noqa: S603 (this interpreter, fixed code)
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert loaded.stdout == "False\nTrue True\n"


def test_a_lazy_name_is_the_defining_modules() -> None:
    from keepout.pc import compute_pc

    assert keepout.compute_pc is compute_pc
    assert "compute_pc" in dir(keepout)
    with pytest.raises(AttributeError, match="has no attribute 'nothing'"):
        _ = keepout.nothing  # type: ignore[attr-defined]
