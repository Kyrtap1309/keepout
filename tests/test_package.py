import importlib
import pkgutil
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
    from keepout.screening import CloseApproach, ScreeningResult, screen
    from keepout.state import StateVector

    assert keepout.Conjunction is Conjunction
    assert keepout.compute_pc is compute_pc
    assert keepout.StateVector is StateVector
    assert keepout.ConjunctionEvent is ConjunctionEvent
    assert keepout.assess is assess
    assert keepout.screen is screen
    assert keepout.CloseApproach is CloseApproach
    assert keepout.ScreeningResult is ScreeningResult
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
