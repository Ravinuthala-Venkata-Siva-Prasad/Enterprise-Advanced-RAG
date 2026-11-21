from src.memory.conversation_store import ConversationStore
from src.memory.conversation_manager import ConversationManager


def main():
    store = ConversationStore(
        db_path="data/conversations/chat_history.db"
    )

    manager = ConversationManager(
        store=store,
        max_turns=5
    )

    session_id = "demo-session-001"

    manager.add_user_message(
        session_id,
        "What is the company MFA policy?"
    )

    manager.add_assistant_message(
        session_id,
        "MFA must be enabled for company email, source-code repositories, and cloud consoles."
    )

    manager.add_user_message(
        session_id,
        "What about remote access?"
    )

    print("\n========== CONVERSATION HISTORY ==========")
    print(manager.format_history(session_id))

    print("\n========== SESSION ISOLATION TEST ==========")
    print("Another session:")
    print(manager.format_history("different-session"))

    print("\nConversation memory smoke test completed.")


if __name__ == "__main__":
    main()