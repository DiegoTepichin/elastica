import elastica


def test_version_is_exposed() -> None:
    assert elastica.__version__
