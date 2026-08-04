"""HTTP client for a running BotBase server.

Talks only to the public API (login, start-chat, chat-profile) so the
simulator has zero code dependency on the main project.
"""

import requests


class BotClientError(RuntimeError):
    pass


class BotClient:
    def __init__(self, base_url, password=None, profile="default", timeout=600):
        self.base_url = base_url.rstrip("/")
        self.profile = profile
        self.timeout = timeout
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
        try:
            resp = self.session.post(
                f"{self.base_url}/api/chat-profile",
                json={
                    "chat_id": self.chat_id,
                    "profile": self.profile,
                    "fullContext": self.full_context,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise BotClientError(f"chat-profile failed: {e}")

        data = resp.json()
        self.full_context = (data.get("messages") or {}).get("full_context") or self.full_context
        reply = (data.get("message") or {}).get("content") or ""
        return reply.strip()

    def list_profiles(self):
        try:
            resp = self.session.get(f"{self.base_url}/api/profiles", timeout=30)
            resp.raise_for_status()
            return resp.json().get("profiles", [])
        except requests.RequestException as e:
            raise BotClientError(f"profiles failed: {e}")
