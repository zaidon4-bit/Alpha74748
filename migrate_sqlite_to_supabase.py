#!/usr/bin/env python3
"""One-time, transactional migration from Alpha AI's local SQLite DB to Supabase."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

try:
    import psycopg
except ImportError as exc:  # pragma: no cover - exercised on user's machine
    raise SystemExit("Missing dependency. Run: python -m pip install 'psycopg[binary]'") from exc

QWEN_MODEL = "qwen/qwen3.7-max:free"
GEMMA_MODEL = "gemma-4-26b-a4b-it"


def pg_dsn() -> str:
    dsn = (os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DB_URL") or "").strip()
    if not dsn:
        raise SystemExit("Set DATABASE_URL to the Supabase PostgreSQL connection string first.")
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://"):]
    if "sslmode=" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return dsn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    found = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return bool(found)


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def value(row: sqlite3.Row, available: set[str], name: str, default=None):
    return row[name] if name in available else default


def migrate(source_path: Path) -> dict[str, int]:
    if not source_path.is_file():
        raise SystemExit(f"SQLite database not found: {source_path}")

    source = sqlite3.connect(source_path)
    source.row_factory = sqlite3.Row
    source.execute("PRAGMA foreign_keys=ON")

    required = {"users", "chat_messages"}
    missing = {name for name in required if not table_exists(source, name)}
    if missing:
        raise SystemExit(f"Source database is missing required tables: {', '.join(sorted(missing))}")

    user_cols = columns(source, "users")
    message_cols = columns(source, "chat_messages")
    source_users = source.execute("SELECT * FROM users ORDER BY id").fetchall()
    source_messages = source.execute("SELECT * FROM chat_messages ORDER BY id").fetchall()
    source_prompts = (
        source.execute("SELECT model_id,prompt,updated_at FROM model_prompts").fetchall()
        if table_exists(source, "model_prompts") else []
    )
    source_routes = (
        source.execute("SELECT * FROM model_api_routes ORDER BY user_id,model_id,priority,id").fetchall()
        if table_exists(source, "model_api_routes") else []
    )
    route_cols = columns(source, "model_api_routes") if table_exists(source, "model_api_routes") else set()

    counts = {"users": 0, "prompts": 0, "messages": 0, "api_routes": 0}
    dsn = pg_dsn()

    try:
        # Do not duplicate chat histories accidentally. Users/admin may already exist
        # because the Render service creates its initial admin and schema on first boot.
        with psycopg.connect(dsn, connect_timeout=15, prepare_threshold=None) as target:
            existing_messages = target.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0]
            if existing_messages:
                raise SystemExit(
                    "Supabase already contains chat messages. Migration stopped to avoid duplicates; "
                    "use a fresh Supabase project/database or review it manually."
                )

            user_id_map: dict[int, int] = {}
            for old in source_users:
                old_id = old["id"]
                username = old["username"]
                existing = target.execute(
                    "SELECT id FROM users WHERE lower(username)=lower(%s)", (username,)
                ).fetchone()

                old_values = {
                    "password_hash": old["password_hash"],
                    "role": old["role"],
                    "api_key_enc": value(old, user_cols, "api_key_enc"),
                    "google_api_key_enc": value(old, user_cols, "google_api_key_enc"),
                    "xkiro_api_key_enc": value(old, user_cols, "xkiro_api_key_enc"),
                    "created_at": value(old, user_cols, "created_at"),
                }
                if existing:
                    new_id = existing[0]
                    target.execute(
                        """UPDATE users SET password_hash=%s, role=%s, api_key_enc=%s,
                           google_api_key_enc=%s, xkiro_api_key_enc=%s,
                           created_at=COALESCE(%s, created_at) WHERE id=%s""",
                        (
                            old_values["password_hash"], old_values["role"],
                            old_values["api_key_enc"], old_values["google_api_key_enc"],
                            old_values["xkiro_api_key_enc"], old_values["created_at"], new_id,
                        ),
                    )
                else:
                    new_id = target.execute(
                        """INSERT INTO users
                           (username,password_hash,role,api_key_enc,google_api_key_enc,xkiro_api_key_enc,created_at)
                           VALUES(%s,%s,%s,%s,%s,%s,COALESCE(%s,CURRENT_TIMESTAMP::text))
                           RETURNING id""",
                        (
                            username, old_values["password_hash"], old_values["role"],
                            old_values["api_key_enc"], old_values["google_api_key_enc"],
                            old_values["xkiro_api_key_enc"], old_values["created_at"],
                        ),
                    ).fetchone()[0]
                user_id_map[int(old_id)] = int(new_id)
                counts["users"] += 1

            for prompt in source_prompts:
                target.execute(
                    """INSERT INTO model_prompts(model_id,prompt,updated_at)
                       VALUES(%s,%s,COALESCE(%s,CURRENT_TIMESTAMP::text))
                       ON CONFLICT(model_id) DO UPDATE SET
                       prompt=EXCLUDED.prompt, updated_at=EXCLUDED.updated_at""",
                    (prompt["model_id"], prompt["prompt"], prompt["updated_at"]),
                )
                counts["prompts"] += 1

            source_route_pairs: set[tuple[int, str]] = set()
            for route in source_routes:
                old_user_id = int(route["user_id"])
                if old_user_id not in user_id_map:
                    continue
                source_route_pairs.add((old_user_id, route["model_id"]))
                target.execute(
                    """INSERT INTO model_api_routes
                       (user_id,model_id,priority,provider,base_url,upstream_model,api_key_enc,created_at)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,COALESCE(%s,CURRENT_TIMESTAMP::text))
                       ON CONFLICT(user_id,model_id,priority) DO NOTHING""",
                    (
                        user_id_map[old_user_id], route["model_id"], route["priority"],
                        route["provider"], value(route, route_cols, "base_url", ""),
                        route["upstream_model"], route["api_key_enc"],
                        value(route, route_cols, "created_at"),
                    ),
                )
                counts["api_routes"] += 1

            # Older SQLite databases may have provider keys but no route-pool table.
            # Convert those keys into the first route so the deployed models remain enabled.
            for old in source_users:
                old_user_id = int(old["id"])
                mapped_user_id = user_id_map.get(old_user_id)
                if mapped_user_id is None:
                    continue
                legacy_routes = (
                    (QWEN_MODEL, "xkiro", "https://api.xkiro.com/v1/chat/completions", QWEN_MODEL,
                     value(old, user_cols, "xkiro_api_key_enc")),
                    (GEMMA_MODEL, "google", "", GEMMA_MODEL,
                     value(old, user_cols, "google_api_key_enc") or value(old, user_cols, "api_key_enc")),
                )
                for model_id, provider, endpoint, upstream_model, encrypted_key in legacy_routes:
                    if encrypted_key and (old_user_id, model_id) not in source_route_pairs:
                        target.execute(
                            """INSERT INTO model_api_routes
                               (user_id,model_id,priority,provider,base_url,upstream_model,api_key_enc)
                               VALUES(%s,%s,1,%s,%s,%s,%s)
                               ON CONFLICT(user_id,model_id,priority) DO NOTHING""",
                            (mapped_user_id, model_id, provider, endpoint, upstream_model, encrypted_key),
                        )
                        counts["api_routes"] += 1

            for message in source_messages:
                old_user_id = int(message["user_id"])
                if old_user_id not in user_id_map:
                    continue
                target.execute(
                    """INSERT INTO chat_messages(user_id,role,parts_json,display_text,created_at)
                       VALUES(%s,%s,%s,%s,COALESCE(%s,CURRENT_TIMESTAMP::text))""",
                    (
                        user_id_map[old_user_id], message["role"], message["parts_json"],
                        message["display_text"], value(message, message_cols, "created_at"),
                    ),
                )
                counts["messages"] += 1

        # Supabase tables use BIGSERIAL for IDs; no manual sequence reset is needed
        # because this importer lets Postgres assign fresh IDs.
    finally:
        source.close()

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Move Alpha AI data from SQLite to Supabase PostgreSQL.")
    parser.add_argument("--source", default="app.db", help="Path to the existing local SQLite database (default: app.db)")
    args = parser.parse_args()
    result = migrate(Path(args.source).expanduser().resolve())
    print("Migration completed successfully:")
    for key, count in result.items():
        print(f"  {key}: {count}")
    print("Keep the original app.db and .api_encryption.key as backups until you verify the deployed app.")


if __name__ == "__main__":
    main()
