"""Events, posted to a URL somebody else owns.

The notification transport, and the whole of it: one class satisfying
:class:`~dashboard.providers.base.Notifier`, one registry entry, two
environment variables. Slack, Discord, ntfy, Zapier and a script behind
``nc -l`` all accept the same thing — a POST with a JSON body — so this speaks
that and has no opinion about which is on the other end.

**One body that three services accept.** Slack reads ``text``, Discord reads
``content``, and everything programmatic wants the fields. Sending all three is
a few dozen bytes and spares this project a "which flavour of webhook" setting,
which is a setting nobody can answer without opening the receiving service's
documentation anyway.

**A transport is not a source.** Nothing here is fetched, cached or rendered;
:mod:`dashboard.services.watch` decides what happened and this decides where it
goes. It is in ``providers/`` because it is chosen from a registry by an
environment variable exactly as the feeds are, and two registries would be one
too many.

**Failure is a ``False``, never an exception.** The live section this is
attached to is about football. A webhook that 404s must not cost a reader the
scores, so the reason is kept in :attr:`error` for a caption and the page
carries on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests
from urllib3.exceptions import LocationParseError
from urllib3.util import parse_url

from dashboard.domain.match import MatchEvent
from src.utils.http import HttpClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

WEBHOOK_URL_ENV = "DASHBOARD_WEBHOOK_URL"
"""Where events go. An environment variable and never ``configs/config.yaml``:
a Slack or Discord webhook URL *is* the credential — anyone holding it can post
as that integration — and that file is committed."""

TIMEOUT_SECONDS = 5.0
"""Short, because this runs inside a page render. A transport that hangs is a
live section that hangs, and the reader is here for the scores rather than for
the delivery."""

MAX_RETRIES = 0
"""None. ``HttpClient`` only retries GET and HEAD anyway — a POST that failed
may already have been acted on — and this is stated rather than inherited so
that a duplicate goal alert can never be this module's doing."""


@dataclass(slots=True)
class WebhookNotifier:
    """One POST per event, to whatever is at :attr:`url`."""

    url: str = field(default_factory=lambda: os.environ.get(WEBHOOK_URL_ENV, ""))
    http: HttpClient | None = None
    """Injected by tests, which mount a stub adapter so no test posts anywhere."""

    name: str = "webhook"
    error: str | None = None

    @property
    def available(self) -> bool:
        """Whether a URL is set. Not whether it works — that costs a request.

        Deliberately not a probe: the only way to ask a webhook whether it is
        healthy is to post to it, and a dashboard that posted a test message on
        every page load would be a dashboard nobody keeps configured.
        """
        if not self.url:
            self.error = (
                f"No webhook configured. Set `{WEBHOOK_URL_ENV}` to post events "
                "to Slack, Discord, ntfy or anything else that accepts a POST."
            )
            return False
        self.error = None
        return True

    def send(self, event: MatchEvent) -> bool:
        """Post one event. ``False`` with the reason in :attr:`error`."""
        if not self.available:
            return False
        client = self.http or HttpClient(timeout_seconds=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
        try:
            client.post(self.url, json=payload(event))
        except requests.RequestException as failure:
            self.error = _describe(failure, self.url)
            logger.info("webhook refused %s: %s", event.kind, self.error)
            return False
        else:
            self.error = None
            return True
        finally:
            # Only a client built here is closed here. An injected one belongs
            # to whoever passed it, and closing another object's connection
            # pool after one call is how the *next* call gets a confusing
            # error — the rule `dashboard/client.py` documents.
            if self.http is None:
                client.close()


def payload(event: MatchEvent) -> dict[str, object]:
    """The body, in the three shapes the common receivers read.

    ``text`` for Slack, ``content`` for Discord, and the event's own fields for
    anything that wants to branch on what happened rather than print a line.
    """
    return {"text": event.message, "content": event.message, **event.as_payload()}


def _describe(failure: requests.RequestException, url: str) -> str:
    """Why the post did not land, in a sentence a caption or a log can carry.

    **Never the URL.** A Slack, Discord or ntfy webhook URL *is* the credential
    — the token is its path — and ``str()`` of a ``requests`` connection error
    embeds the full URL, query string included. So only the host and the
    exception's class survive, which is enough to tell a DNS failure from a
    refused connection without handing the log a working webhook.
    """
    response = failure.response
    if response is not None:
        return f"the webhook answered {response.status_code}: {response.reason or 'no detail'}"
    return f"the webhook at {_host(url)} could not be reached ({type(failure).__name__})"


def _host(url: str) -> str:
    """The host alone — never the path, query or credentials in the URL.

    A URL too malformed to parse is the likeliest reason a post failed, and
    naming it here must not become a second failure.
    """
    try:
        return parse_url(url).host or "the configured host"
    except LocationParseError:
        return "the configured host"
