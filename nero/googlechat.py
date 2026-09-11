"""Talk to Nero from Google Chat, over a Cloud Pub/Sub subscription.

Google Chat delivers to an app one of two ways: it POSTs to an HTTPS endpoint
you host, or it publishes to a Pub/Sub topic you pull from. Only the second
works from a laptop, and Google built it for exactly this case — an app behind
a firewall. Nero pulls; nothing listens for inbound connections.

One turn here is exactly one turn in the terminal: `ChatLoop.ask` runs it, so
the fallback chain, key rotation, memory recall, skills and every error message
behave identically. This module only moves text.

## Why the setup is longer than the others

Telegram, Discord and Slack each hand you a token. Google Chat has no such
thing: an app authenticates as a *service account*, and receiving needs a
Pub/Sub topic and subscription that the Chat API is configured to publish to.
That is three Google Cloud objects before the first message arrives, and none
of it can be automated from here. `nero googlechat setup` says what to make.

Two identifiers are needed alongside the credentials, and neither is a secret,
so they live in config rather than the keyring: the Google Cloud project and
the subscription name.

## Acknowledge, then answer

Pub/Sub redelivers anything unacknowledged, so an event is acked as soon as it
is understood — before the turn runs, which can take 30 seconds. Acking
afterwards would mean Google redelivering the question while Nero is still
answering the first copy.

## The trust boundary

The same one every bridge has, and it lives in `nero/channels.py`: an
allowlist, a pairing code that travels by Google Chat and is approved at the
terminal, and destructive skills refused throughout.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from collections.abc import Callable

import httpx

from nero.channels import (
    PAIRING_REPLY,
    ChannelError,
    PeerStore,
    split,
    starred_markdown,
)

logger = logging.getLogger("nero.googlechat")

CHAT_ROOT = "https://chat.googleapis.com/v1"
PUBSUB_ROOT = "https://pubsub.googleapis.com/v1"
KEYRING_ENTRY = "googlechat_service_account"

# Posting as the app needs chat.bot; pulling the subscription needs pubsub.
# Asked for together because one credential does both jobs.
SCOPES = (
    "https://www.googleapis.com/auth/chat.bot",
    "https://www.googleapis.com/auth/pubsub",
)

# Google Chat's ceiling on one message's text.
GOOGLE_CHAT_LIMIT = 4096

HTTP_TIMEOUT = 30.0
# Pub/Sub's REST pull returns as soon as anything is waiting and otherwise
# holds the request briefly, so an idle bridge is a slow loop rather than a
# busy one. This is the gap between empty pulls.
IDLE_SECONDS = 2.0
RETRY_SECONDS = 5
MAX_MESSAGES = 10


class GoogleChatError(ChannelError):
    """Google refused the credentials, or could not be reached."""


class GoogleChatBot:
    """Pulling events from Pub/Sub and posting replies to Chat.

    Both are plain REST with a bearer token; the only thing google-auth is here
    for is minting and refreshing that token from the service-account key.
    """

    def __init__(
        self,
        credentials_json: str,
        project_id: str,
        subscription_id: str,
        client: httpx.Client | None = None,
    ):
        self.project_id = project_id
        self.subscription_id = subscription_id
        self._client = client or httpx.Client(timeout=HTTP_TIMEOUT)
        self._credentials = _credentials(credentials_json)

    @property
    def subscription(self) -> str:
        return f"projects/{self.project_id}/subscriptions/{self.subscription_id}"

    def _token(self) -> str:
        """A live access token, refreshed when it has expired.

        Refresh is checked per call rather than scheduled: a bridge can sit idle
        for hours past an hour-long token's life, and a timer would have to
        outlive that anyway.
        """
        from google.auth.transport.requests import Request

        try:
            if not self._credentials.valid:
                self._credentials.refresh(Request())
        except Exception as exc:  # noqa: BLE001 — google-auth raises broadly
            raise GoogleChatError(
                f"Google would not issue a token for that service account: {exc}"
            ) from exc
        return self._credentials.token

    def _call(self, method: str, url: str, **body):
        try:
            response = self._client.request(
                method,
                url,
                headers={"Authorization": f"Bearer {self._token()}"},
                json=body or None,
            )
        except httpx.HTTPError as exc:
            raise GoogleChatError(f"Could not reach Google: {exc}") from exc
        if response.status_code in (401, 403):
            raise GoogleChatError(_why(response))
        if response.status_code >= 400:
            raise GoogleChatError(_why(response))
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise GoogleChatError("Google sent a response Nero could not read.") from exc

    def username(self) -> str:
        """The service account this is running as.

        Chat has no getMe; the account's own email is the useful thing to show,
        and proves the credentials parse.
        """
        return getattr(self._credentials, "service_account_email", "?")

    def check(self) -> None:
        """Prove the credentials work and the subscription exists.

        Done at startup rather than on the first message: a wrong project or a
        subscription that was never created is otherwise a bridge that runs
        quietly forever and answers nothing.
        """
        self._call("GET", f"{PUBSUB_ROOT}/{self.subscription}")

    def pull(self, max_messages: int = MAX_MESSAGES) -> list[tuple[str, dict]]:
        """(ack_id, event) for each waiting message.

        A message whose payload is not readable JSON is returned with an empty
        event so the caller still acknowledges it — otherwise one malformed
        publish is redelivered forever and blocks everything behind it.
        """
        payload = self._call(
            "POST", f"{PUBSUB_ROOT}/{self.subscription}:pull", maxMessages=max_messages
        )
        received = []
        for message in payload.get("receivedMessages") or []:
            ack_id = message.get("ackId")
            if not ack_id:
                continue
            raw = (message.get("message") or {}).get("data")
            try:
                event = json.loads(base64.b64decode(raw or "")) if raw else {}
            except (ValueError, TypeError):
                logger.debug("undecodable Pub/Sub payload; acknowledging it anyway")
                event = {}
            received.append((ack_id, event if isinstance(event, dict) else {}))
        return received

    def acknowledge(self, ack_ids: list[str]) -> None:
        if ack_ids:
            self._call(
                "POST", f"{PUBSUB_ROOT}/{self.subscription}:acknowledge", ackIds=ack_ids
            )

    def send(self, space: str, text: str) -> None:
        for part in split(to_google_chat(text), GOOGLE_CHAT_LIMIT):
            self._call("POST", f"{CHAT_ROOT}/{space}/messages", text=part)

    def close(self) -> None:
        self._client.close()


def _credentials(credentials_json: str):
    from google.oauth2 import service_account

    try:
        info = json.loads(credentials_json)
    except ValueError as exc:
        raise GoogleChatError(
            "That service account key is not valid JSON. Download it again from "
            "the Google Cloud console."
        ) from exc
    try:
        return service_account.Credentials.from_service_account_info(info, scopes=list(SCOPES))
    except (ValueError, KeyError) as exc:
        raise GoogleChatError(f"That service account key is missing something: {exc}") from exc


def _why(response: httpx.Response) -> str:
    """Google's own words for a failure, with the ones worth explaining
    explained. Its 403s in particular say almost nothing on their own."""
    try:
        message = (response.json().get("error") or {}).get("message", "")
    except ValueError:
        message = ""
    if response.status_code == 403 and "pubsub" in response.request.url.host:
        return (
            f"Google refused the subscription ({message or '403'}). Give the "
            "service account the Pub/Sub Subscriber role on it."
        )
    if response.status_code == 403:
        return (
            f"Google refused the request ({message or '403'}). Check the Chat "
            "API is enabled and the app is published to your workspace."
        )
    if response.status_code == 401:
        return "Google rejected the service account. Re-run `nero googlechat setup`."
    if response.status_code == 404:
        return (
            f"Google could not find that ({message or '404'}). Check the project "
            "and subscription names in `nero config show`."
        )
    return message or f"Google refused the request ({response.status_code})."


def to_google_chat(text: str) -> str:
    """`text` in Google Chat's formatting.

    The same `*bold*` / `_italic_` dialect Slack reads, minus the escaping:
    Google Chat does not treat `&`, `<` or `>` as markup in ordinary text, and
    escaping them would show entities to the reader.
    """
    return starred_markdown(text, str)


def incoming(event: dict) -> tuple[str, str] | None:
    """(space, text) from a Chat event, or None if it is not a turn.

    Filtered out here: everything that is not a MESSAGE (Chat publishes
    ADDED_TO_SPACE, REMOVED_FROM_SPACE and card clicks on the same topic),
    anything sent by a bot — including Nero's own replies, without which it
    answers itself forever — and anything with no text.
    """
    if event.get("type") not in (None, "MESSAGE"):
        return None
    message = event.get("message") or {}
    if (message.get("sender") or {}).get("type") == "BOT":
        return None
    space = (message.get("space") or {}).get("name") or (event.get("space") or {}).get("name")
    text = (message.get("text") or "").strip()
    if not space or not text:
        return None
    return str(space), text


def serve(
    bot: GoogleChatBot,
    allowed_space_ids: set[str],
    ask: Callable[[str], str | None],
    on_event: Callable[[str], None] = lambda _m: None,
    once: bool = False,
    peers: PeerStore | None = None,
    allow_pairing: bool = True,
    refresh: Callable[[], set[str]] | None = None,
    stop: threading.Event | None = None,
) -> None:
    """Pull the subscription and answer messages until interrupted.

    `ask` is `ChatLoop.ask`. An unpaired space is offered a pairing code (see
    `nero/channels.py`) and nothing else; `allow_pairing=False` restores plain
    silence for anyone running with the door shut.

    `refresh` re-reads the allowlist as messages arrive. Without it a space
    approved while this is running stays locked out until the bridge is
    restarted — you approve, Google Chat keeps getting pairing codes, and
    nothing says why.
    """
    if not allowed_space_ids and not allow_pairing:
        raise GoogleChatError(
            "No space is allowed to message this app yet. Run `nero googlechat setup` first."
        )
    if peers is None:
        peers = PeerStore("googlechat")
    # Once, before the loop: a subscription that was never created, or one the
    # service account cannot read, is not a transient failure. Retrying it
    # every five seconds forever is how a bridge looks alive and answers
    # nothing.
    bot.check()

    while stop is None or not stop.is_set():
        try:
            received = bot.pull()
        except GoogleChatError as exc:
            on_event(f"Google Chat is unreachable ({exc}). Retrying.")
            if once:
                return
            time.sleep(RETRY_SECONDS)
            continue

        # Before any turn runs: a turn can take 30 s and Pub/Sub redelivers
        # well inside that, which would have Nero answer the same question
        # twice. Every id is acked, including the ones that turn out not to be
        # turns, or they come back forever.
        try:
            bot.acknowledge([ack_id for ack_id, _event in received])
        except GoogleChatError as exc:
            logger.debug("could not acknowledge: %s", exc)

        for _ack_id, event in received:
            allowed_space_ids = _answer(
                bot, event, allowed_space_ids, ask, on_event, peers, allow_pairing, refresh,
            )
        if once:
            return
        if not received and (stop is None or not stop.is_set()):
            time.sleep(IDLE_SECONDS)


def _answer(
    bot: GoogleChatBot,
    event: dict,
    allowed: set[str],
    ask: Callable[[str], str | None],
    on_event: Callable[[str], None],
    peers: PeerStore,
    allow_pairing: bool,
    refresh: Callable[[], set[str]] | None,
) -> set[str]:
    """Handle one Chat event. Returns the allowlist to use from here on, which
    `refresh` may have just widened."""
    received = incoming(event)
    if received is None:
        return allowed
    space, text = received
    if refresh is not None:
        allowed = refresh()
    if space not in allowed:
        # The message itself is never run. All an unpaired space can get is a
        # code, which is worth nothing without terminal access.
        logger.warning("pairing offered to space %s (not allowed)", space)
        if not allow_pairing:
            on_event(f"Ignored a message from {space} (not paired).")
            return allowed
        code = peers.request(space)
        try:
            bot.send(space, PAIRING_REPLY.format(code=code, channel="googlechat"))
        except GoogleChatError as exc:
            logger.debug("could not offer pairing: %s", exc)
        # The code is deliberately absent here: it has to travel by Google Chat,
        # or approving it proves nothing about who is holding the account.
        on_event(
            f"{space} asked to pair. Approve with the code shown in Google "
            "Chat: nero googlechat approve <code>"
        )
        return allowed
    on_event(f"{space}: {text}")
    try:
        reply = ask(text)
    except Exception as exc:  # noqa: BLE001 — the bridge must outlive a bad turn
        logger.debug("turn failed", exc_info=True)
        reply = f"Something went wrong with that turn: {exc}"
    try:
        bot.send(space, reply or "(no reply)")
    except GoogleChatError as exc:
        on_event(f"Could not reply in {space}: {exc}")
    return allowed
