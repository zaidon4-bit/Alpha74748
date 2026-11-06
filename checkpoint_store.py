"""Validation helpers for Alpha's user-owned project file checkpoints.

This module validates generated text files before they are persisted. It never
extracts archives, evaluates source, or executes user-provided code.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any

MAX_CHECKPOINT_FILES = 120
MAX_CHECKPOINT_FILE_BYTES = 2 * 1024 * 1024
MAX_CHECKPOINT_TOTAL_BYTES = 8 * 1024 * 1024
MAX_CHECKPOINT_LABEL_CHARS = 120
MAX_CHECKPOINT_TASK_CHARS = 800

_SECRET_EXACT_NAMES = {
    ".env", ".env.local", ".env.development", ".env.production", ".env.test",
    ".api_encryption.key", ".flask_secret_key", "id_rsa", "id_ed25519",
    "credentials.json", "app.db", "service-account.json", "secrets.json", "token.json",
    ".npmrc", ".pypirc", ".netrc", "id_rsa", "id_ed25519", "known_hosts", "authorized_keys",
}
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks")
_SECRET_DIRECTORIES = {".git", ".ssh", ".aws", ".config", ".cache", ".gnupg", "secrets", "private", "credentials"}
_SAFE_SAMPLE_ENV = {".env.example", ".env.sample", ".env.template"}


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint contains unsafe or oversized data."""


def _safe_path(value: Any) -> str:
    if not isinstance(value, str):
        raise CheckpointValidationError("مسار الملف غير صالح.")
    path = value.strip()
    if not path or len(path) > 240 or "\\" in path or path.startswith("/"):
        raise CheckpointValidationError("مسار الملف غير صالح.")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts) or ":" in parts[0]:
        raise CheckpointValidationError("مسار الملف غير آمن.")
    posix = PurePosixPath(path)
    if posix.is_absolute() or any(part.lower() in _SECRET_DIRECTORIES for part in parts):
        raise CheckpointValidationError("مجلدات الحالة والأسرار غير مسموحة في نقاط الاستعادة.")
    lower_name = posix.name.lower()
    if lower_name in _SECRET_EXACT_NAMES or lower_name.endswith(_SECRET_SUFFIXES):
        raise CheckpointValidationError("لا يمكن حفظ ملف أسرار ضمن نقطة الاستعادة.")
    if lower_name.startswith(".env.") and lower_name not in _SAFE_SAMPLE_ENV:
        raise CheckpointValidationError("ملفات البيئة الحساسة غير مسموحة في نقاط الاستعادة.")
    return posix.as_posix()


def normalize_checkpoint_files(raw_files: Any) -> list[dict[str, str]]:
    """Validate a list of source files and return normalized path/content/language."""
    if not isinstance(raw_files, list) or not raw_files:
        raise CheckpointValidationError("لا توجد ملفات برمجية لحفظ نقطة الاستعادة.")
    if len(raw_files) > MAX_CHECKPOINT_FILES:
        raise CheckpointValidationError(f"عدد الملفات يتجاوز الحد ({MAX_CHECKPOINT_FILES}).")

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    total_bytes = 0
    for item in raw_files:
        if not isinstance(item, dict):
            raise CheckpointValidationError("صيغة أحد الملفات غير صالحة.")
        path = _safe_path(item.get("path"))
        if path in seen:
            raise CheckpointValidationError("تحتوي نقطة الاستعادة على مسارات ملفات مكررة.")
        seen.add(path)
        content = item.get("content")
        if not isinstance(content, str):
            raise CheckpointValidationError(f"محتوى الملف {path} غير صالح.")
        size = len(content.encode("utf-8"))
        if size > MAX_CHECKPOINT_FILE_BYTES:
            raise CheckpointValidationError(f"الملف {path} أكبر من الحد المسموح.")
        total_bytes += size
        if total_bytes > MAX_CHECKPOINT_TOTAL_BYTES:
            raise CheckpointValidationError("الحجم الإجمالي لنقطة الاستعادة يتجاوز 8 ميغابايت.")
        language = item.get("language", "Text")
        if not isinstance(language, str):
            language = "Text"
        result.append({"path": path, "content": content, "language": language[:40]})

    return result


def checkpoint_digest(files: list[dict[str, str]]) -> str:
    """Return a stable digest so accidental duplicate snapshots can be avoided."""
    payload = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_checkpoint_label(value: Any, task: Any, files: list[dict[str, str]]) -> str:
    label = value.strip() if isinstance(value, str) else ""
    if not label:
        task_line = task.strip().splitlines()[0] if isinstance(task, str) and task.strip() else ""
        label = task_line or files[0]["path"]
    label = re.sub(r"\s+", " ", label).strip()
    return (label or "Project checkpoint")[:MAX_CHECKPOINT_LABEL_CHARS]


def normalize_checkpoint_task(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()[:MAX_CHECKPOINT_TASK_CHARS]
