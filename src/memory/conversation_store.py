import sqlite3
from pathlib import Path
from typing import Any


class ConversationStore:
    """SQLite-backed storage for isolated conversation sessions."""

    def __init__(self, db_path: str = "data/conversations/chat_history.db"):
        self.db_path = db_path

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self):
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_messages
                ON messages(session_id, id)
            """)

    def add_message(self, session_id: str, role: str, content: str):
        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty")

        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("content cannot be empty")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                (session_id, role, content.strip())
            )

    def get_messages(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit)
            ).fetchall()

        # Return chronological order.
        return [dict(row) for row in reversed(rows)]

    def clear_session(self, session_id: str):
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,)
            )