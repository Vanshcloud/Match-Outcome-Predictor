"""Who the reader is, decided in one place.

Milestone 14. Favourites used to belong to a browser tab; they now belong to
*someone*, and this module is the only thing in the application that knows how
that someone is named. :mod:`dashboard.domain.store` keys on whatever
:func:`current` returns and has no opinion about where it came from.

**Two ways to be someone, and the second one is optional.**

*A profile* — a name picked from the sidebar. No password, no provider, no
network. This is the default, and it is honest about what it is: a profile is
not an account, and the docs say so. Anyone who can open the page can pick any
profile, which is the correct amount of security for something
``docs/DASHBOARD.md`` describes as not for public hosting.

*An account* — Streamlit's own OIDC login, when the deployment configures a
provider. The key is then the identity provider's verified email rather than
anything typed into this application.

**How "is a provider configured" is known, without touching secrets.**
Streamlit adds ``is_logged_in`` to ``st.user`` only when ``secrets.toml`` has an
``[auth]`` section — with no section the object has no attributes at all, and
reading ``st.secrets`` directly raises when there is no file. So the *presence*
of the key is the signal, read through ``.get`` so that it is never an
``AttributeError`` on a machine that has never heard of OIDC.

**The two namespaces cannot collide.** A key is ``account:<email>`` or
``profile:<name>``, never a bare string. Without the prefixes, someone who
typed the email address of a logged-in colleague as their profile name would be
handed that colleague's favourites — a small thing here, and exactly the shape
of a serious thing in an application that stored more.
"""

from __future__ import annotations

import importlib.util

import streamlit as st

PROFILE_KEY = "profile"
"""Which profile this tab is using. The only thing left in ``st.session_state``.

A choice, not data: the favourites themselves are in the store, and this says
which of them to read. It stays per-tab on purpose — two tabs open on two
profiles is a reasonable thing to want, and it is what a "who's watching"
picker implies.
"""

GUEST = "Guest"
"""The profile a reader who has picked nothing is using.

A real profile with a real row in the store rather than a null case, so that
favourites persist for someone who never opens the picker — which is the whole
user-visible point of this milestone.
"""

ACCOUNT_PREFIX = "account:"
PROFILE_PREFIX = "profile:"

NEW_PROFILE = "＋ New profile…"
"""The picker's last entry, which is a request rather than a name.

Here beside :data:`GUEST` rather than in the sidebar that renders it, so that a
test can name it without importing ``dashboard/app.py`` — that module calls
``main()`` at import, and a test that imported it would render the whole
application as a side effect of reading one string.
"""


def signed_in() -> bool:
    """Whether an identity provider has authenticated this reader."""
    return bool(st.user.get("is_logged_in"))


def provider_configured() -> bool:
    """Whether this deployment has an ``[auth]`` section at all.

    Streamlit adds the ``is_logged_in`` key only then, so its presence — True
    *or* False — is the answer, and its absence means no provider.
    """
    return st.user.get("is_logged_in") is not None


def sign_in_offered() -> bool:
    """Whether the sidebar should show a sign-in button.

    Both halves, because ``st.login()`` raises ``StreamlitMissingAuthlibError``
    when ``streamlit[auth]`` is not installed — and this image does not install
    it, deliberately. A button that always errors is worse than no button, so
    the sidebar shows what the deployment can actually do.
    """
    return provider_configured() and importlib.util.find_spec("authlib") is not None


def account() -> str | None:
    """The signed-in reader's verified email, or ``None`` when nobody is."""
    if not signed_in():
        return None
    email = st.user.get("email")
    return str(email) if email else None


def profile() -> str:
    """The profile this tab has chosen."""
    return str(st.session_state.get(PROFILE_KEY) or GUEST)


def use_profile(name: str) -> str:
    """Switch this tab to ``name``. Returns the profile actually in use.

    Trimmed, and an empty name is :data:`GUEST` rather than a profile whose
    name is a space — the picker takes free text, so this is where a person's
    typing meets a storage key.
    """
    chosen = " ".join(name.split()) or GUEST
    st.session_state[PROFILE_KEY] = chosen
    return chosen


def current() -> str:
    """The store key for this reader: an account if there is one, else a profile."""
    email = account()
    return f"{ACCOUNT_PREFIX}{email}" if email else f"{PROFILE_PREFIX}{profile()}"


def label() -> str:
    """How the reader is named on screen."""
    return account() or profile()


def display(key: str) -> str:
    """A store key as a person reads it. The inverse of :func:`current`.

    Used by the picker, which lists what the store holds and must not show
    ``profile:Vansh`` to somebody.
    """
    for prefix in (ACCOUNT_PREFIX, PROFILE_PREFIX):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def profile_names(keys: list[str]) -> list[str]:
    """The profile names among ``keys``, for the picker.

    Accounts are deliberately excluded: they are not a thing anyone chooses
    from a dropdown, and listing them would put the email addresses of everyone
    who has ever signed in on an unauthenticated page.

    :data:`GUEST` and the profile currently in use are always included, whether
    or not the store has heard of them. A profile is created by being *chosen*,
    and it has no row until it has a favourite — so without this, a reader who
    had just named themselves would not find their own name in the picker.
    """
    names = [display(key) for key in keys if key.startswith(PROFILE_PREFIX)]
    return sorted(set(names) | {GUEST, profile()})
