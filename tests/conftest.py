import pytest

from halo.fixtures import make_fixture


@pytest.fixture(scope="session")
def fixture_default():
    return make_fixture(seed=0)
