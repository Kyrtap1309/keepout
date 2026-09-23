import keepout


def test_version_matches_distribution_metadata() -> None:
    from importlib.metadata import version

    assert keepout.__version__ == version("keepout")
