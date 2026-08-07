"""HTTP client for a running BotBase server.

Talks only to the public API (login, start-chat, chat-profile) so the
simulator has zero code dependency on the main project.

Turns run through the bot's async flow (chat-profile with async=true,
then polling get-messages / turn-result) so no single HTTP request lasts
longer than a poll — which survives reverse proxies with short read
timeouts. A pre-async BotBase server that just runs the turn inline and
returns 200 with the full payload is still handled.
"""

import time

import requests


class BotClientError(RuntimeError):
    pass


class BotClient:
    def __init__(self, base_url, password=None, profile="default", timeout=600,
                 arch=None):
        self.base_url = base_url.rstrip("/")
        self.profile = profile
        self.timeout = timeout
        # Agent architecture ('single' / 'multi'). None means "don't send
        # the field" — the bot server falls back to its own default, and
        # pre-multi-agent servers never see an unknown key.
        self.arch = arch or None
        self.session = requests.Session()
        self.chat_id = None
        self.full_context = []
        if password:
            self._login(password)

    def _login(self, password):
        try:
            resp = self.session.post(
                f"{self.base_url}/login", json={"password": password}, timeout=30)
        except requests.RequestException as e:
            raise BotClientError(f"Cannot reach bot server at {self.base_url}: {e}")
        if resp.status_code != 200:
            raise BotClientError(f"Bot server login failed (HTTP {resp.status_code})")

    def start_chat(self):
        try:
            resp = self.session.post(f"{self.base_url}/api/start-chat", timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise BotClientError(f"start-chat failed: {e}")
        self.chat_id = resp.json().get("chat_id")
        if not self.chat_id:
            raise BotClientError("start-chat returned no chat_id")
        self.full_context = []
        return self.chat_id

    def send(self, user_message):
        """Send one user message; block until the bot's turn completes.

        Returns the bot's user-visible reply text for the turn.
        """
        if not self.chat_id:
            self.start_chat()
        self.full_context.append({"role": "user", "content": user_message})
        payload = {
            "chat_id": self.chat_id,
            "profile": self.profile,
            "fullContext": self.full_context,
            "async": True,
        }
        if self.arch:
            payload["arch"] = self.arch
        try:
            # self.timeout (not a short one) so a pre-async server that runs
            # the whole turn inside this request still gets its time.
            resp = self.session.post(
                f"{self.base_url}/api/chat-profile",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise BotClientError(f"chat-profile failed: {e}")

        data = resp.json()
        if resp.status_code == 202 or data.get("accepted"):
            data = self._await_turn()
        # else: pre-async server already returned the full payload.

        if data.get("error"):
            raise BotClientError(f"bot turn failed: {data['error']}")
        self.full_context = (data.get("messages") or {}).get("full_context") or self.full_context
        reply = (data.get("message") or {}).get("content") or ""
        return reply.strip()

    def _await_turn(self):
        """Poll an async turn to completion and fetch its final payload."""
        deadline = time.time() + self.timeout
        complete = False
        while time.time() < deadline:
            try:
                r = self.session.get(
                    f"{self.base_url}/api/get-messages/{self.chat_id}",
                    timeout=30,
                )
                r.raise_for_status()
                if r.json().get("is_complete"):
                    complete = True
                    break
            except requests.RequestException:
                pass  # transient; bounded by the deadline
            time.sleep(1.0)
        if not complete:
            raise BotClientError(
                f"bot turn did not complete within {self.timeout}s")

        # The result is stashed a beat after is_complete flips; retry briefly.
        for _ in range(20):
            try:
                r = self.session.get(
                    f"{self.base_url}/api/turn-result/{self.chat_id}",
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                if data.get("ready"):
                    return data
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise BotClientError("bot turn completed but its result never became ready")

    def list_profiles(self):
        try:
            resp = self.session.get(f"{self.base_url}/api/profiles", timeout=30)
            resp.raise_for_status()
            return resp.json().get("profiles", [])
        except requests.RequestException as e:
            raise BotClientError(f"profiles failed: {e}")

    def get_architectures(self):
        """Agent architectures the bot server supports, its default, and
        the server's router profile key (the entry-point profile a real
        user lands on).

        Returns {"architectures": [...], "default": str|None,
        "router": str|None}. A BotBase server from before the multi-agent
        split has no arch fields in /api/topics — treat that as 'no
        choice to offer'."""
        try:
            resp = self.session.get(f"{self.base_url}/api/topics", timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return {
                "architectures": data.get("architectures") or [],
                "default": data.get("arch_default"),
                "router": data.get("router"),
            }
        except requests.RequestException as e:
            raise BotClientError(f"topics failed: {e}")
