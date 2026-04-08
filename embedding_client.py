"""
Client for the embedding microservice.

Provides the interface expected by search_resources / examine_resource tools:
  - switch_provider(provider)
  - search(queries, language, k) -> {resource_id: score}
  - get_resource_details(resource_id) -> dict | None
"""

import os
import sqlite3

import requests


class EmbeddingSearchClient:
    def __init__(self, service_url: str = "http://localhost:8200",
                 resources_root: str = "./processed_resources"):
        self.service_url = service_url.rstrip("/")
        self.resources_root = resources_root
        self._provider: str | None = None
        self._db_conn: sqlite3.Connection | None = None

    def switch_provider(self, provider: str):
        """Set the active provider (database / resource directory name)."""
        if provider == self._provider:
            return
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None
        self._provider = provider
        db_path = os.path.join(self.resources_root, provider, "database.db")
        if os.path.exists(db_path):
            self._db_conn = sqlite3.connect(db_path)
            self._db_conn.row_factory = sqlite3.Row

    def search(self, queries: list[str], language: str = "all", k: int = 10) -> dict[str, float]:
        """
        Hybrid search via the embedding microservice.
        Returns {resource_id: score} (higher is better).
        """
        if not self._provider:
            raise RuntimeError("No provider set — call switch_provider() first")

        resp = requests.post(
            f"{self.service_url}/search",
            json={
                "provider": self._provider,
                "queries": queries,
                "k": k,
                "language": language,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["resource_scores"]

    def get_resource_details(self, resource_id: int) -> dict | None:
        """Fetch full resource row from the local SQLite database."""
        if self._db_conn is None:
            return None
        try:
            cursor = self._db_conn.execute(
                "SELECT * FROM resources WHERE id = ?", (resource_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)
        except sqlite3.Error:
            return None

    def close(self):
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None
