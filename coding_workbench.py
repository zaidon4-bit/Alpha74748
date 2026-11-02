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
        "خريطة المشروع: هذه الأداة تعمل حتى دون أي مرفق. إذا أرفق المستخدم ملفات أو أرشيف مشروع، حلّل ما أتيح لك فعلياً "
        "ولخّص البنية والمجلدات ونقاط الدخول والمكونات والاعتماديات وتدفق البيانات، مع التمييز بين الحقائق والاستنتاجات. "
        "إذا لم توجد ملفات، أنشئ خريطة مهمة مرئية اعتماداً على نص المستخدم وسياق المحادثة: الهدف والمراحل والعلاقات والمخرجات. "
        "لا تطلب رفع ملف كشرط للبدء، ولا تدّعِ أنك فحصت مستودعاً أو ملفات غير متاحة."
    ),
    "impact_analysis": (
        "تحليل تأثير التعديل: اعمل من وصف المستخدم وسياق المحادثة حتى دون مرفقات. قبل اقتراح أي تغيير، حدّد مناطق النظام "
        "والوظائف والواجهات والبيانات التي قد تتأثر، واذكر الافتراضات وما يحتاج تحققاً من الملفات لاحقاً. "
        "لا تخترع أسماء ملفات أو دوال غير معروفة، وحافظ على السلوك والميزات السابقة ولا توسّع نطاق التعديل."
    ),
    "test_plan": (
        "خطة الاختبارات: أنشئ خطة اختبار من وصف المهمة حتى دون إرفاق المشروع، تغطي النجاح والفشل والحالات الحدّية. "
        "عندما تكون بيئة المشروع أو ملفاته غير متاحة، اذكر أوامر تحقق عامة أو افتراضية بوضوح ولا تخترع نتائج. "
        "لا تنفّذ الأوامر ولا تدّعِ أن الاختبارات نُفذت؛ صنّفها بوضوح على أنها مقترحة أو غير مشغّلة."
    ),
    "patch_review": (
        "مراجعة التغييرات: اعمل من الطلب النصي حتى دون ملفات. اقترح أصغر تغيير مطلوب، واشرح المناطق المتأثرة. "
        "قدّم Diff فعلياً فقط عند توفر نص الملفات؛ وإلا قدّم تصوراً للتغيير دون اختراع محتوى ملفات أو مسارات. "
        "لا تدّعِ أنك عدّلت ملفات المستخدم أو طبّقت Patch تلقائياً."
    ),
    "log_doctor": (
        "طبيب السجلات: اعمل حتى دون سجلات مرفقة. إذا وُجدت سجلات، ابدأ بأول خطأ سببي واربط الاستنتاج بالنص والأسطر. "
        "إذا لم توجد، استخدم وصف المستخدم لبناء مسار تشخيص مرتب وحدد ما يلزم جمعه لاحقاً دون إيقاف المساعدة. "
        "ميّز السبب المؤكد عن الاحتمالات، ولا تكرر كلمات المرور أو مفاتيح API أو الرموز السرية؛ استبدلها بعلامة [مخفي]."
    ),
    "security_review": (
        "مراجعة الأمان: اعمل من وصف المستخدم حتى دون ملفات. عند غياب الشيفرة، قدّم مراجعة معمارية مبدئية وافصلها "
        "عن النتائج المثبتة بالملفات. عند توفر الشيفرة، راجع المصادقة والصلاحيات والمدخلات والأسرار والجلسات والسجلات. "
        "لا تنسب ثغرة إلى ملف غير متاح، واقترح إصلاحات دفاعية فقط."
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
        "[ALPHA AUTOMATIC CODING WORKBENCH — CURRENT TURN ONLY]\n"
        "Apply these selected modes to the user's current message and conversation context even when no file is attached. "
        "Attachments are optional evidence, not a prerequisite. If files are absent, still provide the useful conceptual "
        "analysis, plan, relationships, and next steps; state only which file-specific facts cannot be verified. "
        "These are user-facing workflows, not background processes. Do not execute shell commands, run uploaded code, "
        "install packages, modify files, or apply patches automatically.\n\n"
        + "\n".join("- " + CODING_TOOL_GUIDANCE[key] for key in selected)
    )


def build_structured_response_instruction(enabled: bool) -> str:
    """Require a rendered task-plan card whenever structured planning/tools are enabled."""
    if not enabled:
        return ""
    return (
        "[ALPHA STRUCTURED RESPONSE CARD — REQUIRED FOR THIS TURN]\n"
        "Alongside your normal helpful Arabic answer, finish with exactly one fenced block labelled alpha-plan. "
        "The block must contain valid JSON only, using this schema: "
        '{"title":"عنوان قصير","summary":"ملخص واضح","stages":[{"title":"اسم المرحلة","details":"ما الذي سيحدث","deliverable":"الناتج المتوقع"}]}. '
        "Use 1–6 stages. This block is rendered by Alpha as a visual plan card, not shown as raw code. "
        "Build the plan from the current message and available conversation context; no upload is required. "
        "For ordinary conversation, keep it brief and natural (one stage is enough when appropriate); do not invent project files, "
        "repository structure, test results, or completed actions. If no project files are attached, describe a task/workflow map, "
        "not a repository file map. Keep all card values in Arabic."
    )
