"""Safe, in-memory project checks and patch application for Alpha.

No uploaded source is executed and no archive is extracted to disk.
"""
from __future__ import annotations

import ast
import difflib
import io
import json
import re
import stat
import zipfile
from html.parser import HTMLParser
from pathlib import PurePosixPath

MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_ENTRIES = 1200
MAX_TOTAL_UNCOMPRESSED = 48 * 1024 * 1024
MAX_MEMBER_BYTES = 4 * 1024 * 1024
MAX_PATCH_BYTES = 512 * 1024
SKIP_PARTS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build"}
SECRET_NAMES = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519", "credentials.json", ".api_encryption.key", ".flask_secret_key", "app.db"}


class ProjectArchiveError(ValueError):
    pass


class _HTMLCounter(HTMLParser):
    def error(self, message):  # pragma: no cover - HTMLParser keeps permissive parsing
        raise ValueError(message)


def _safe_member_name(name: str) -> bool:
    if not isinstance(name, str) or not name or "\\" in name or name.startswith("/"):
        return False
    path = PurePosixPath(name)
    return not any(part in ("..", "") for part in path.parts) and not (path.parts and ":" in path.parts[0])


def _open_checked_archive(raw: bytes):
    if not isinstance(raw, (bytes, bytearray)) or not raw or len(raw) > MAX_ARCHIVE_BYTES:
        raise ProjectArchiveError("حجم أرشيف المشروع غير صالح أو يتجاوز الحد المسموح.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
        infos = archive.infolist()
    except (zipfile.BadZipFile, OSError) as exc:
        raise ProjectArchiveError("الملف المرفوع ليس أرشيف ZIP صالحاً.") from exc
    if len(infos) > MAX_ENTRIES:
        archive.close()
        raise ProjectArchiveError("عدد الملفات داخل الأرشيف يتجاوز الحد الآمن.")
    total = 0
    seen = set()
    for info in infos:
        if not _safe_member_name(info.filename):
            archive.close()
            raise ProjectArchiveError("يحتوي الأرشيف على مسار ملف غير آمن.")
        if info.filename in seen:
            archive.close()
            raise ProjectArchiveError("يحتوي الأرشيف على أسماء ملفات مكررة.")
        seen.add(info.filename)
        if info.file_size > MAX_MEMBER_BYTES:
            archive.close()
            raise ProjectArchiveError(f"الملف {PurePosixPath(info.filename).name} أكبر من الحد الآمن.")
        total += info.file_size
        if total > MAX_TOTAL_UNCOMPRESSED:
            archive.close()
            raise ProjectArchiveError("الحجم بعد فك الضغط يتجاوز الحد الآمن.")
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            archive.close()
            raise ProjectArchiveError("الأرشيف يحتوي على رابط رمزي غير مسموح.")
    return archive, infos


def static_check_project_zip(raw: bytes) -> dict:
    """Run syntax/format checks only. Never imports or executes project code."""
    archive, infos = _open_checked_archive(raw)
    checks = []
    skipped = []
    scanned = 0
    try:
        for info in infos:
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            if any(part in SKIP_PARTS for part in path.parts) or path.name in SECRET_NAMES:
                skipped.append(path.as_posix())
                continue
            suffix = path.suffix.lower()
            if suffix not in {".py", ".pyw", ".json", ".html", ".htm"}:
                continue
            if info.file_size > 1024 * 1024:
                skipped.append(path.as_posix() + " (أكبر من حد الفحص)")
                continue
            scanned += 1
            try:
                text = archive.read(info).decode("utf-8-sig")
                if suffix in {".py", ".pyw"}:
                    ast.parse(text, filename=path.as_posix())
                    checks.append({"file": path.as_posix(), "kind": "Python", "ok": True, "message": "صياغة Python سليمة."})
                elif suffix == ".json":
                    json.loads(text)
                    checks.append({"file": path.as_posix(), "kind": "JSON", "ok": True, "message": "بنية JSON سليمة."})
                else:
                    parser = _HTMLCounter(convert_charrefs=True)
                    parser.feed(text)
                    parser.close()
                    checks.append({"file": path.as_posix(), "kind": "HTML", "ok": True, "message": "تم تحليل HTML دون أخطاء بنيوية من المحلل."})
            except (UnicodeDecodeError, SyntaxError, json.JSONDecodeError, ValueError) as exc:
                checks.append({"file": path.as_posix(), "kind": suffix.lstrip("."), "ok": False,
                               "message": f"{type(exc).__name__}: {str(exc)[:220]}"})
    finally:
        archive.close()
    passed = sum(1 for item in checks if item["ok"])
    failed = sum(1 for item in checks if not item["ok"])
    return {
        "status": "completed", "mode": "static_only", "scanned": scanned,
        "passed": passed, "failed": failed, "skipped": len(skipped),
        "checks": checks[:250], "skipped_files": skipped[:100],
        "notice": "هذا فحص ساكن للصياغة والتنسيقات فقط؛ لم يتم تشغيل أو استيراد أي كود من الأرشيف.",
    }


def _strip_git_prefix(path: str) -> str:
    path = path.strip()
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _patchable_path(path: str) -> bool:
    """Allow edits only to ordinary project files, never secrets or runtime state."""
    if not _safe_member_name(path):
        return False
    parsed = PurePosixPath(path)
    if any(part in SKIP_PARTS for part in parsed.parts):
        return False
    name = parsed.name.lower()
    if name in SECRET_NAMES or name.startswith(".env"):
        return False
    if name.endswith((".pem", ".key", ".p12", ".pfx")):
        return False
    return True


def _parse_patch(patch_text: str):
    if not isinstance(patch_text, str) or not patch_text.strip():
        raise ProjectArchiveError("أدخل Unified Diff صالحاً أولاً.")
    if len(patch_text.encode("utf-8")) > MAX_PATCH_BYTES:
        raise ProjectArchiveError("حجم التعديل أكبر من الحد المسموح.")
    lines = patch_text.splitlines(keepends=True)
    patches = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("diff --git "):
            i += 1
            continue
        if not lines[i].startswith("--- "):
            i += 1
            continue
        old_name = lines[i][4:].strip().split("\t", 1)[0]
        i += 1
        if i >= len(lines) or not lines[i].startswith("+++ "):
            raise ProjectArchiveError("ترويسة Unified Diff غير مكتملة.")
        new_name = lines[i][4:].strip().split("\t", 1)[0]
        i += 1
        if old_name == "/dev/null" or new_name == "/dev/null":
            raise ProjectArchiveError("إنشاء الملفات أو حذفها عبر Diff غير مدعوم في هذه المرحلة.")
        old_path, new_path = _strip_git_prefix(old_name), _strip_git_prefix(new_name)
        if old_path != new_path or not _patchable_path(old_path):
            raise ProjectArchiveError("التعديل يحاول تغيير مسار الملف أو الوصول إلى ملف حساس/غير مسموح.")
        hunks = []
        while i < len(lines) and lines[i].startswith("@@ "):
            match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[i])
            if not match:
                raise ProjectArchiveError("تعذر قراءة عنوان جزء من التعديل.")
            old_start, old_count, new_start, new_count = match.groups()
            old_count = int(old_count) if old_count is not None else 1
            new_count = int(new_count) if new_count is not None else 1
            i += 1
            hunk_lines = []
            old_seen = new_seen = 0
            while i < len(lines) and (old_seen < old_count or new_seen < new_count):
                line = lines[i]
                if line.startswith("\\ No newline"):
                    i += 1
                    continue
                if not line or line[0] not in " +-":
                    raise ProjectArchiveError("محتوى جزء التعديل غير مكتمل أو غير قياسي.")
                marker = line[0]
                hunk_lines.append(line)
                if marker in " -":
                    old_seen += 1
                if marker in " +":
                    new_seen += 1
                i += 1
            if old_seen != old_count or new_seen != new_count:
                raise ProjectArchiveError("عدد الأسطر في جزء التعديل لا يطابق العنوان.")
            hunks.append({
                "old_start": int(old_start), "old_count": old_count,
                "new_start": int(new_start), "new_count": new_count,
                "lines": hunk_lines,
            })
        if not hunks:
            raise ProjectArchiveError("لم يتم العثور على أجزاء تعديل قابلة للتطبيق.")
        patches.append({"path": old_path, "hunks": hunks})
    if not patches:
        raise ProjectArchiveError("لم يتم العثور على ملفات ضمن Unified Diff.")
    paths = [p["path"] for p in patches]
    if len(paths) != len(set(paths)):
        raise ProjectArchiveError("التعديل يحتوي على أكثر من كتلة لنفس الملف؛ اجمعها في كتلة واحدة.")
    return patches


def _apply_file_hunks(original: bytes, hunks: list[dict]) -> bytes:
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProjectArchiveError("لا يمكن تطبيق Diff على ملف ثنائي أو غير UTF-8.") from exc
    source = text.splitlines(keepends=True)
    output = []
    cursor = 0
    offset = 0
    for hunk in hunks:
        target = max(0, hunk["old_start"] - 1 + offset)
        if target < cursor or target > len(source):
            raise ProjectArchiveError("ترتيب أجزاء التعديل غير صالح.")
        output.extend(source[cursor:target])
        cursor = target
        old_seen = new_seen = 0
        for item in hunk["lines"]:
            marker, payload = item[0], item[1:]
            if marker in " -":
                if cursor >= len(source):
                    raise ProjectArchiveError("محتوى الملف لا يطابق سياق التعديل.")
                current = source[cursor]
                if current.rstrip("\r\n") != payload.rstrip("\r\n"):
                    raise ProjectArchiveError("التعديل لا يطابق محتوى الملف الحالي؛ لم يتم تغيير الأرشيف.")
                old_seen += 1
                if marker == " ":
                    output.append(current)
                    new_seen += 1
                cursor += 1
            elif marker == "+":
                output.append(payload)
                new_seen += 1
        if old_seen != hunk["old_count"] or new_seen != hunk["new_count"]:
            raise ProjectArchiveError("أعداد الأسطر في عنوان Diff لا تطابق محتواه.")
        offset += new_seen - old_seen
    output.extend(source[cursor:])
    return "".join(output).encode("utf-8")


def prepare_patched_archive(raw_zip: bytes, patch_text: str):
    patches = _parse_patch(patch_text)
    archive, infos = _open_checked_archive(raw_zip)
    info_by_name = {info.filename: info for info in infos if not info.is_dir()}
    missing = [p["path"] for p in patches if p["path"] not in info_by_name]
    if missing:
        archive.close()
        raise ProjectArchiveError("التعديل يشير إلى ملفات غير موجودة في الأرشيف: " + ", ".join(missing[:8]))
    try:
        originals = {p["path"]: archive.read(info_by_name[p["path"]]) for p in patches}
        updated = {p["path"]: _apply_file_hunks(originals[p["path"]], p["hunks"]) for p in patches}
        preview = []
        for path in updated:
            before = originals[path].decode("utf-8").splitlines()
            after = updated[path].decode("utf-8").splitlines()
            diff = list(difflib.unified_diff(before, after, fromfile="قبل/" + path, tofile="بعد/" + path, lineterm=""))
            preview.append({"file": path, "added": sum(1 for line in diff if line.startswith("+") and not line.startswith("+++")),
                            "removed": sum(1 for line in diff if line.startswith("-") and not line.startswith("---")),
                            "diff": "\n".join(diff[:160])})
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as rebuilt:
            for info in infos:
                data = updated.get(info.filename)
                if data is None:
                    data = archive.read(info) if not info.is_dir() else b""
                rebuilt.writestr(info, data)
        return output.getvalue(), preview
    finally:
        archive.close()
