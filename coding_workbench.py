"""User-selected, advisory coding-workbench modes for Alpha AI.

This module only validates feature flags and builds prompt guidance. It does not
execute code, commands, tests, or modify project files.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence

CODING_TOOL_GUIDANCE = {
    "project_map": (
        "خريطة المشروع: حلّل ملفات المشروع التي أرفقها المستخدم فقط. لخّص البنية والمجلدات ونقاط الدخول "
        "والمكونات والاعتماديات وتدفق البيانات. ميّز بوضوح بين ما وجدته فعلاً وما لا يمكن استنتاجه. "
        "إذا لم توجد ملفات مشروع مرفقة، اطلب رفع الأرشيف أو الملفات ولا تدّعِ فحص مستودع غير متاح."
    ),
    "impact_analysis": (
        "تحليل تأثير التعديل: قبل اقتراح أي تغيير، حدّد الملفات والوظائف والواجهات وقاعدة البيانات التي قد تتأثر، "
        "واذكر مخاطر التوافق والانحدار. حافظ على السلوك والميزات السابقة، ولا توسّع نطاق التعديل إلى ملفات غير لازمة."
    ),
    "test_plan": (
        "خطة الاختبارات: اقترح اختبارات محددة لحالات النجاح والفشل والحالات الحدّية، واكتب أوامر التحقق المناسبة "
        "لبيئة المشروع. لا تنفّذ الأوامر ولا تدّعِ أن الاختبارات نُفذت؛ صنّفها بوضوح على أنها مقترحة أو غير مشغّلة."
    ),
    "patch_review": (
        "مراجعة التغييرات: اجعل التغييرات المقترحة صغيرة ومحصورة في المطلوب. اذكر الملفات المتأثرة وسبب كل تغيير، "
        "وقدّم Diff واضحاً عند توفر سياق كافٍ. لا تدّعِ أنك عدّلت ملفات المستخدم أو طبّقت Patch تلقائياً."
    ),
    "log_doctor": (
        "طبيب السجلات: عند وجود سجلات أو Traceback مرفقة، ابدأ بأول خطأ سببي لا بآخر أثر تابع له، "
        "واربط الاستنتاج بالرسائل والأسطر الموجودة. ميّز السبب المؤكد عن الاحتمالات، ولا تكرر كلمات المرور أو "
        "مفاتيح API أو الرموز السرية إن ظهرت في السجلات؛ استبدلها بعلامة [مخفي]."
    ),
    "security_review": (
        "مراجعة الأمان: راجع المصادقة والصلاحيات والتحقق من المدخلات وتخزين الأسرار والجلسات وسجلات الأخطاء "
        "وحدود المعدل. اذكر الثغرات مع دليل من الملفات، واقترح إصلاحات دفاعية فقط."
    ),
}


def parse_coding_tool_flags(raw_value: object) -> list[str]:
    """Accept only the small allowlisted set of user-selected workbench modes."""
    if isinstance(raw_value, str):
        try:
            raw_value = json.loads(raw_value)
        except (TypeError, ValueError):
            return []
    if not isinstance(raw_value, (list, tuple)):
        return []
    selected: list[str] = []
    for value in raw_value:
        if isinstance(value, str) and value in CODING_TOOL_GUIDANCE and value not in selected:
            selected.append(value)
    return selected



def normalize_review_issues(raw_issues: object, blocks: Sequence[dict], min_confidence: float = 0.82) -> list[dict]:
    """Keep only concrete, high-confidence findings tied to an exact code excerpt.

    This filters vague, duplicate, block-mismatched, and speculative model warnings.
    It does not execute code or decide that an API is safe/unsafe by name alone.
    """
    if not isinstance(raw_issues, list):
        return []

    code_by_index = {
        block.get("index"): block.get("code", "")
        for block in blocks
        if isinstance(block, dict)
        and isinstance(block.get("index"), int)
        and not isinstance(block.get("index"), bool)
        and isinstance(block.get("code", ""), str)
    }
    normalized: list[dict] = []
    seen: set[tuple[int, str, str]] = set()

    for item in raw_issues[:40]:
        if not isinstance(item, dict):
            continue
        block_index = item.get("block")
        if (not isinstance(block_index, int) or isinstance(block_index, bool)
                or block_index not in code_by_index):
            continue

        severity = str(item.get("severity", "warning")).strip().lower()
        if severity == "critical":
            severity = "error"
        if severity not in {"error", "warning"}:
            continue

        description = item.get("description")
        evidence = item.get("evidence")
        impact = item.get("impact")
        suggested_fix = item.get("suggested_fix", "")
        if not all(isinstance(value, str) for value in (description, evidence, impact)):
            continue
        description = re.sub(r"\s+", " ", description).strip()
        evidence = evidence.strip()
        impact = re.sub(r"\s+", " ", impact).strip()
        suggested_fix = re.sub(r"\s+", " ", suggested_fix).strip() if isinstance(suggested_fix, str) else ""
        code = code_by_index[block_index]

        # The reviewer must quote a literal excerpt from the cited block.
        if len(description) < 12 or len(evidence) < 8 or evidence not in code or len(impact) < 12:
            continue
        if severity == "error" and not suggested_fix:
            continue

        raw_confidence = item.get("confidence", 0.0)
        if isinstance(raw_confidence, bool):
            continue
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            continue
        if not 0.0 <= confidence <= 1.0 or confidence < min_confidence:
            continue

        key = (
            block_index,
            re.sub(r"\s+", " ", evidence).casefold(),
            re.sub(r"\s+", " ", description).casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        normalized.append({
            "block": block_index,
            "severity": severity,
            "description": description[:1200],
            "evidence": evidence[:500],
            "impact": impact[:800],
            "confidence": confidence,
            "suggested_fix": suggested_fix[:1200],
        })
        if len(normalized) >= 20:
            break

    return normalized


def split_review_findings(issues: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Separate confirmed errors (eligible for repair) from non-blocking advisories."""
    blocking = [item for item in issues if isinstance(item, dict) and item.get("severity") == "error"]
    advisories = [item for item in issues if isinstance(item, dict) and item.get("severity") == "warning"]
    return blocking, advisories

def build_coding_tool_prompt_section(selected_tools: Sequence[str]) -> str:
    """Build subordinate current-turn instructions for the selected modes."""
    selected = parse_coding_tool_flags(list(selected_tools))
    if not selected:
        return ""
    return (
        "[USER-SELECTED OPTIONAL CODING WORKBENCH — CURRENT TURN ONLY]\n"
        "Apply only these selected analysis modes to the current request. They are advisory workflows, "
        "not background processes. Do not execute shell commands, run uploaded code, install packages, "
        "modify files, or apply patches automatically. Ask for missing project files when necessary.\n\n"
        + "\n".join("- " + CODING_TOOL_GUIDANCE[key] for key in selected)
    )
