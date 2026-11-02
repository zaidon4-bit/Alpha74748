"""User-selected, advisory coding-workbench modes for Alpha AI.

This module only validates feature flags and builds prompt guidance. It does not
execute code, commands, tests, or modify project files.
"""
from __future__ import annotations

import json
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
