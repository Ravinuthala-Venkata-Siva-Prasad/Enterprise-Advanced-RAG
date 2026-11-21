class ConversationManager:
    def __init__(self, store, max_turns: int = 10):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        self.store = store
        self.max_messages = max_turns * 2

    def add_user_message(self, session_id: str, message: str):
        self.store.add_message(session_id, "user", message)

    def add_assistant_message(self, session_id: str, message: str):
        self.store.add_message(session_id, "assistant", message)

    def get_history(self, session_id: str):
        return self.store.get_messages(
            session_id=session_id,
            limit=self.max_messages
        )

    def format_history(self, session_id: str) -> str:
        history = self.get_history(session_id)

        if not history:
            return "No previous conversation."

        return "\n".join(
            f"{item['role'].upper()}: {item['content']}"
            for item in history
        )

    def clear_session(self, session_id: str):
        self.store.clear_session(session_id)