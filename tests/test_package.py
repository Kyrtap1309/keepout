import keepout


def test_version_matches_distribution_metadata() -> None:
    from importlib.metadata import version

    assert keepout.__version__ == version("keepout")


def test_core_api_is_importable_from_the_package() -> None:
    from keepout.conjunction import Conjunction
    from keepout.pc import compute_pc
    from keepout.state import StateVector

    assert keepout.Conjunction is Conjunction
    assert keepout.compute_pc is compute_pc
    assert keepout.StateVector is StateVector
    assert all(hasattr(keepout, name) for name in keepout.__all__)
