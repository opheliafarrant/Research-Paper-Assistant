"""Conversation state and operations, independent of storage and the paper library."""

from collections.abc import MutableMapping
from uuid import uuid4


def new_conversation(state: MutableMapping) -> dict:
    conversation = {
        "id": uuid4().hex,
        "title": "新对话",
        "histories": {},
        "mode": "论文问答",
        "paper_title": None,
        "summaries": {},
    }
    state.setdefault("conversations", []).append(conversation)
    state["active_conversation_id"] = conversation["id"]
    return conversation


def active_conversation(state: MutableMapping) -> dict:
    conversations = state.setdefault("conversations", [])
    active_id = state.get("active_conversation_id")
    for conversation in conversations:
        if conversation["id"] == active_id:
            return conversation
    return new_conversation(state)


def select_conversation(state: MutableMapping, conversation_id: str) -> None:
    if not any(item["id"] == conversation_id for item in state.get("conversations", [])):
        raise ValueError("Conversation does not exist")
    state["active_conversation_id"] = conversation_id


def delete_conversation(state: MutableMapping, conversation_id: str) -> None:
    """Delete a conversation by ID. If all conversations are removed, create a fresh one."""
    conversations = state.setdefault("conversations", [])
    if not any(item["id"] == conversation_id for item in conversations):
        raise ValueError("Conversation does not exist")
    state["conversations"] = [item for item in conversations if item["id"] != conversation_id]
    if not state["conversations"]:
        new_conversation(state)
    elif state.get("active_conversation_id") == conversation_id:
        state["active_conversation_id"] = state["conversations"][-1]["id"]


def rename_conversation(conversation: dict, new_title: str) -> bool:
    """Rename a conversation with a cleaned, bounded title."""
    clean = " ".join(new_title.strip().split())
    if not clean:
        return False
    conversation["title"] = clean[:30]
    return True


def history_for(conversation: dict, mode: str, paper_id: str = "") -> list:
    key = f"{mode}:{paper_id}"
    return conversation["histories"].setdefault(key, [])


def summary_for(conversation: dict, paper_id: str) -> dict | None:
    return conversation.setdefault("summaries", {}).get(paper_id)


def set_summary(conversation: dict, paper_id: str, text: str, aspect: str) -> None:
    conversation.setdefault("summaries", {})[paper_id] = {"text": text, "aspect": aspect}


def clear_summary_for(conversation: dict, paper_id: str) -> None:
    conversation.setdefault("summaries", {}).pop(paper_id, None)


def name_from_question(conversation: dict, question: str) -> None:
    if conversation["title"] == "新对话":
        compact = " ".join(question.split())
        conversation["title"] = compact[:20] + ("…" if len(compact) > 20 else "")


def mode_icon(mode: str) -> str:
    """Material icon corresponding to conversation scope."""
    return ":material/article:" if mode == "单篇精读" else ":material/chat:"


def conversation_stats(conversation: dict) -> dict:
    """Summarize the message and structured extraction counts of a conversation."""
    histories = conversation.get("histories", {})
    total_messages = sum(len(msgs) for msgs in histories.values())
    summaries_count = len(conversation.get("summaries", {}))
    return {"messages": total_messages, "summaries": summaries_count}

