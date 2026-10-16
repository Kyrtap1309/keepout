import socket
import urllib.request

import pytest
from pytest_socket import SocketBlockedError


def test_a_test_cannot_open_a_network_socket() -> None:
    # pytest-socket warns as it refuses; the suite turns warnings into errors.
    with (
        pytest.warns(UserWarning, match=r"socket\.socket"),
        pytest.raises(SocketBlockedError),
    ):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_a_test_cannot_download() -> None:
    with pytest.raises(AssertionError, match="must not access the network"):
        urllib.request.urlopen("https://example.org/")
