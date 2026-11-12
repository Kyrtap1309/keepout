"""The Space-Track account: credentials from the environment, and the
redaction that keeps them out of messages."""

from __future__ import annotations

import os
import re
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self

from keepout._types import override
from keepout.errors import AuthError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "HIDDEN",
    "IDENTITY_VARIABLE",
    "PASSWORD_VARIABLE",
    "Credentials",
    "redacted_excerpt",
]

IDENTITY_VARIABLE: Final = "SPACETRACK_IDENTITY"
"""The environment variable holding the account's user name."""

# The name of the variable, not a password.
PASSWORD_VARIABLE: Final = "SPACETRACK_PASSWORD"  # noqa: S105
"""The environment variable holding the account's password."""

HIDDEN: Final = "<hidden>"
"""What stands for a credential in text shown to the user."""


@dataclass(frozen=True, slots=True, repr=False)
class Credentials:
    """A Space-Track account's user name and password.

    Neither is shown by ``repr`` or ``str``.

    Parameters
    ----------
    identity
        The user name (the e-mail address the account was opened with).
    password
        The password.

    Raises
    ------
    ValueError
        If either is empty.
    """

    identity: str
    password: str

    def __post_init__(self) -> None:
        if not self.identity or not self.password:
            raise ValueError("Space-Track identity and password must not be empty")

    @override
    def __repr__(self) -> str:
        return f"Credentials({HIDDEN})"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        """The credentials in :data:`IDENTITY_VARIABLE` and
        :data:`PASSWORD_VARIABLE`.

        Parameters
        ----------
        environ
            The environment; :data:`os.environ` if ``None``.

        Raises
        ------
        AuthError
            If a variable is not set or empty (the message names it).
        """
        if environ is None:
            environ = os.environ
        missing = [
            name
            for name in (IDENTITY_VARIABLE, PASSWORD_VARIABLE)
            if not environ.get(name)
        ]
        if missing:
            raise AuthError(
                f"no Space-Track credentials: set {' and '.join(missing)} "
                "(an account is free at https://www.space-track.org)"
            )
        return cls(environ[IDENTITY_VARIABLE], environ[PASSWORD_VARIABLE])

    def redact(self, text: str) -> str:
        """``text`` with the identity and password, as typed or URL-encoded,
        replaced by ``<hidden>``."""
        # Longest first, so a secret inside another is not left half-shown.
        forms = sorted(_secret_forms(self), key=len, reverse=True)
        return re.sub("|".join(map(re.escape, forms)), HIDDEN, text)


def redacted_excerpt(credentials: Credentials, text: str, length: int) -> str:
    """The first ``length`` characters of ``text``, credentials hidden.

    The whole text is redacted before it is cut, so a secret crossing the
    cut is hidden whole; a cut text that ends in the start of a secret (as
    :meth:`Credentials.redact` spells it) loses that end too.
    """
    excerpt = credentials.redact(text)[:length]
    if len(text) <= length:
        return excerpt
    dangling = max(
        (
            size
            for form in _secret_forms(credentials)
            for size in range(1, len(form))
            if excerpt.endswith(form[:size])
        ),
        default=0,
    )
    return excerpt[: len(excerpt) - dangling]


def _secret_forms(credentials: Credentials) -> set[str]:
    """The identity and password as typed, path- and form-encoded."""
    return {
        form
        for secret in (credentials.identity, credentials.password)
        for form in (
            secret,
            urllib.parse.quote(secret, safe=""),
            urllib.parse.quote_plus(secret),
        )
    }
