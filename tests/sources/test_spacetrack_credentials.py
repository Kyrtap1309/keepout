import urllib.parse

import pytest
from spacetrack_builders import CREDENTIALS, IDENTITY, PASSWORD, assert_hidden

from keepout.errors import AuthError
from keepout.sources.spacetrack import (
    HIDDEN,
    IDENTITY_VARIABLE,
    PASSWORD_VARIABLE,
    Credentials,
)


def test_credentials_come_from_the_environment() -> None:
    environ = {IDENTITY_VARIABLE: IDENTITY, PASSWORD_VARIABLE: PASSWORD}
    assert Credentials.from_env(environ) == CREDENTIALS


def test_credentials_default_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(IDENTITY_VARIABLE, IDENTITY)
    monkeypatch.setenv(PASSWORD_VARIABLE, PASSWORD)
    assert Credentials.from_env() == CREDENTIALS


@pytest.mark.parametrize(
    ("environ", "missing"),
    [
        ({}, f"{IDENTITY_VARIABLE} and {PASSWORD_VARIABLE}"),
        ({IDENTITY_VARIABLE: IDENTITY}, PASSWORD_VARIABLE),
        ({IDENTITY_VARIABLE: "", PASSWORD_VARIABLE: PASSWORD}, IDENTITY_VARIABLE),
    ],
)
def test_missing_credentials_name_the_variables(
    environ: dict[str, str], missing: str
) -> None:
    with pytest.raises(AuthError, match=f"set {missing} ") as caught:
        Credentials.from_env(environ)
    assert_hidden(str(caught.value))


def test_credentials_are_not_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Credentials(IDENTITY, "")


def test_credentials_are_not_shown() -> None:
    assert repr(CREDENTIALS) == "Credentials(<hidden>)"
    assert str(CREDENTIALS) == "Credentials(<hidden>)"


def test_credentials_are_redacted_in_any_spelling() -> None:
    text = (
        f"user {IDENTITY} password {PASSWORD} form "
        f"{urllib.parse.urlencode({'identity': IDENTITY, 'password': PASSWORD})} "
        f"path {urllib.parse.quote(PASSWORD, safe='')}"
    )
    redacted = CREDENTIALS.redact(text)
    assert_hidden(redacted)
    assert redacted == (
        "user <hidden> password <hidden> form identity=<hidden>&password=<hidden> "
        "path <hidden>"
    )


def test_a_secret_inside_another_is_hidden_whole() -> None:
    credentials = Credentials("ab", "xaby")
    assert credentials.redact("xaby ab") == f"{HIDDEN} {HIDDEN}"
