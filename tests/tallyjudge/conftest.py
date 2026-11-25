import pytest

from tallyjudge.sdk.testing import ScriptedClient


@pytest.fixture
def client():
    return ScriptedClient()
