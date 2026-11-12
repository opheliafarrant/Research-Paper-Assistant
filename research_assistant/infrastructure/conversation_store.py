"""Local, transactional storage for independently versioned conversations."""

import json
import sqlite3
from contextlib import closing

from research_assistant import config

DB_PATH = config.DATA_DIR / "conversations.sqlite3"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, data TEXT NOT NULL, version INTEGER NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return connection


def load():
    with closing(_connect()) as db, db:
        rows = db.execute("SELECT id, data, version FROM conversations ORDER BY rowid").fetchall()
        setting = db.execute("SELECT value FROM settings WHERE key = 'active_id'").fetchone()
    conversations = []
    versions = {}
    for conversation_id, data, version in rows:
        conversation = json.loads(data)
        if not isinstance(conversation, dict) or conversation.get("id") != conversation_id:
            raise ValueError(f"Invalid stored conversation: {conversation_id}")
        conversations.append(conversation)
        versions[conversation_id] = version
    return conversations, versions, setting[0] if setting else None


def encode(conversation):
    return json.dumps(conversation, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(conversation, version):
    data = encode(conversation)
    with closing(_connect()) as db, db:
        if version is None:
            db.execute("INSERT INTO conversations (id, data, version) VALUES (?, ?, 1)", (conversation["id"], data))
            return 1
        cursor = db.execute("UPDATE conversations SET data = ?, version = version + 1 WHERE id = ? AND version = ?",
                            (data, conversation["id"], version))
        if cursor.rowcount != 1:
            raise RuntimeError("会话已在其他窗口修改或删除，请先保存当前内容并重新加载。")
        return version + 1


def delete_many(versions):
    with closing(_connect()) as db, db:
        for conversation_id, version in versions.items():
            cursor = db.execute("DELETE FROM conversations WHERE id = ? AND version = ?", (conversation_id, version))
            if cursor.rowcount != 1:
                raise RuntimeError("会话已在其他窗口修改或删除，请先保存当前内容并重新加载。")


def set_active(conversation_id):
    with closing(_connect()) as db, db:
        db.execute("INSERT INTO settings (key, value) VALUES ('active_id', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (conversation_id,))
