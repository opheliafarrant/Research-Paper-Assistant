"""Bridge browser session state and the local conversation store."""

from research_assistant.infrastructure import conversation_store as store


def initialize(state):
    if state.get("conversations_loaded"):
        return
    saved, versions, active_id = store.load()
    existing = state.get("conversations", [])
    by_id = {item["id"]: item for item in saved}
    for item in existing:
        by_id[item["id"]] = item
    state["conversations"] = list(by_id.values())
    state["conversation_versions"] = versions
    state["conversation_saved_data"] = {item["id"]: store.encode(item) for item in saved}
    if not state.get("active_conversation_id"):
        state["active_conversation_id"] = active_id
    state["conversations_loaded"] = True
    for item in existing:
        save(state, item)


def save(state, conversation):
    conversation_id = conversation["id"]
    data = store.encode(conversation)
    if state["conversation_saved_data"].get(conversation_id) == data:
        return
    version = store.save(conversation, state["conversation_versions"].get(conversation_id))
    state["conversation_versions"][conversation_id] = version
    state["conversation_saved_data"][conversation_id] = data


def delete(state, conversation_ids):
    versions = {item_id: state["conversation_versions"][item_id] for item_id in conversation_ids}
    store.delete_many(versions)
    for item_id in conversation_ids:
        state["conversation_versions"].pop(item_id, None)
        state["conversation_saved_data"].pop(item_id, None)


def select(conversation_id):
    store.set_active(conversation_id)
