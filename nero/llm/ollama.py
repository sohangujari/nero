"""Helpers for the local Ollama server (no API key — it's all localhost)."""

import subprocess

import httpx

BASE_URL = "http://localhost:11434"


def reachable(base_url: str = BASE_URL, timeout: float = 1.0) -> bool:
    """Is the Ollama server up at its default local address?"""
    try:
        return httpx.get(f"{base_url}/api/version", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def list_models(base_url: str = BASE_URL) -> list[str]:
    """Names of locally pulled models (tags like 'qwen3:8b', 'phi4-mini:latest')."""
    response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
    response.raise_for_status()
    return [model["name"] for model in response.json().get("models", [])]


def has_model(name: str, base_url: str = BASE_URL) -> bool:
    """Whether Ollama would actually serve `name`, the way Ollama resolves it.

    A bare name means the `:latest` tag — it is not a wildcard over whatever
    happens to be pulled. Matching it as a prefix said yes to `llama3.2` on a
    machine holding only `llama3.2:1b`, while Ollama itself answered
    `model 'llama3.2' not found`; a fallback chain built on that answer looks
    like a safety net until the moment it is needed.
    """
    try:
        tags = list_models(base_url)
    except httpx.HTTPError:
        return False
    wanted = name if ":" in name else f"{name}:latest"
    return any(tag == name or tag == wanted for tag in tags)


def supports_tools(name: str, base_url: str = BASE_URL) -> bool | None:
    """Whether the model can do tool calling, per Ollama's own capability list.

    /api/show reports capabilities authoritatively (e.g. phi4-mini has 'tools',
    gemma3 does not), so there's no need to hand-maintain a list or send a probe
    chat. Returns None when it can't be determined — server down, or model not
    pulled — so callers can stay quiet rather than warn on a guess.
    """
    try:
        response = httpx.post(f"{base_url}/api/show", json={"model": name}, timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return "tools" in (response.json().get("capabilities") or [])


# Models Ollama reports as tool-capable that cannot decide when *not* to call
# a tool. This is a different question from the one /api/show answers, and
# Ollama has no field for it: llama3.2 lists "tools" in its capabilities and
# genuinely emits well-formed, schema-valid tool calls — for every message,
# including "hi" and "what is 2+2". Those calls pass validation and execute, so
# no amount of downstream guarding recovers the answer.
#
# Measured on llama3.2:latest (3.2B), temperature 0 and 0.7, offering 12 tools,
# 3 tools and 1 tool: a tool fired on 7 of 7 ordinary messages at every tool
# count ("what is 2+2" -> fetch_web_page; with one tool offered, open_app).
# With no tools offered the same model answers all 7 correctly in ~0.4 s.
#
# Entries are measured, never guessed — a wrong entry silently disables every
# skill for someone whose model works fine. Keyed on the family, before the
# tag, because the behaviour is the model's, not the quantisation's.
MISFIRES_TOOLS = frozenset({"llama3.2"})


def misfires_tools(name: str) -> bool:
    """Whether `name` is a model measured to call tools on ordinary chat.

    Purely a table lookup, so it stays true with the server down — unlike
    `supports_tools`, there is nothing to probe.
    """
    return name.split(":")[0] in MISFIRES_TOOLS


# Asking a small model "does this need a tool?" is a much easier question than
# asking it to decide mid-generation whether to emit a tool call, and it turns
# out they can answer it. Measured on llama3.2 over 20 messages, ten
# conversational and ten asking for an action: 20/20 correct, ~0.3 s each on a
# ~145 token prompt. The same model, offered the same tools, called one on 7 of
# 7 ordinary messages.
#
# Kept deliberately concrete and short. The two labels are words rather than
# yes/no because a model that has been asked a yes/no question will sometimes
# answer the *message* instead of the question.
ACTION_GATE = """Reply with one word: ACT or TALK.

ACT  = the user is asking you to do something on their computer (open or close an
       app, play or pause music, change the volume, open a website, read a file,
       run a command) or asking for live information you would need to look up
       (weather, news, prices, search).
TALK = anything else: greetings, chat, maths, jokes, explanations, or questions
       about the user that you already know.

Message: {message}
Answer with ACT or TALK only."""

# Four tokens is enough for "ACT" or "TALK" and stops a chatty model narrating.
GATE_PREDICT = 4
GATE_TIMEOUT = 20.0


def wants_action(name: str, message: str, base_url: str = BASE_URL) -> bool | None:
    """Whether `message` is asking for something to be done, or just talk.

    None when the question could not be put — server down, model gone, an answer
    that is neither word. The caller then offers tools as before: a wrong tool
    call is visible and recoverable, while withholding tools from a real request
    makes a small model *narrate* the action it did not take, which is worse
    because nothing looks wrong.
    """
    try:
        response = httpx.post(
            f"{base_url}/api/chat",
            json={
                "model": name,
                "messages": [
                    {"role": "user", "content": ACTION_GATE.format(message=message)}
                ],
                "stream": False,
                # No thinking. The budget below is four tokens, and a thinking
                # model spends them on reasoning and answers with an empty
                # string — which is why this gate scored 0/10 against qwen3.5
                # and gemma4 until it said so explicitly.
                "think": False,
                # Temperature 0: this is a classification, and there is nothing
                # to be gained from sampling a different answer to it.
                "options": {"temperature": 0, "num_predict": GATE_PREDICT},
            },
            timeout=GATE_TIMEOUT,
        )
        response.raise_for_status()
        answer = (response.json()["message"].get("content") or "").strip().upper()
    except (httpx.HTTPError, KeyError, ValueError):
        return None
    if "ACT" in answer:
        return True
    if "TALK" in answer:
        return False
    return None


def pull_model(name: str) -> bool:
    """Run `ollama pull`, inheriting stdio so its progress bars render."""
    try:
        return subprocess.run(["ollama", "pull", name]).returncode == 0
    except OSError:
        return False
