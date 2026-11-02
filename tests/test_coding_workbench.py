import unittest

from coding_workbench import (
    CODING_TOOL_GUIDANCE,
    build_coding_tool_prompt_section,
    normalize_review_issues,
    parse_coding_tool_flags,
    split_review_findings,
)


class CodingWorkbenchTests(unittest.TestCase):
    def test_defaults_to_no_tools(self):
        self.assertEqual(parse_coding_tool_flags(None), [])
        self.assertEqual(parse_coding_tool_flags("not-json"), [])
        self.assertEqual(build_coding_tool_prompt_section([]), "")

    def test_accepts_json_encoded_selection(self):
        selected = parse_coding_tool_flags('["project_map", "test_plan"]')
        self.assertEqual(selected, ["project_map", "test_plan"])

    def test_rejects_unknown_values_and_duplicates(self):
        selected = parse_coding_tool_flags([
            "project_map", "project_map", "unknown", None, 7, "security_review"
        ])
        self.assertEqual(selected, ["project_map", "security_review"])

    def test_all_known_modes_have_guidance(self):
        self.assertEqual(
            set(CODING_TOOL_GUIDANCE),
            {"project_map", "impact_analysis", "test_plan", "patch_review", "log_doctor", "security_review"},
        )
        self.assertEqual(len(parse_coding_tool_flags(list(CODING_TOOL_GUIDANCE))), 6)

    def test_prompt_section_is_optional_and_non_executing(self):
        section = build_coding_tool_prompt_section(["test_plan", "patch_review"])
        self.assertIn("CURRENT TURN ONLY", section)
        self.assertIn("لا تنفّذ الأوامر", section)
        self.assertIn("لا تدّعِ", section)
        self.assertNotIn("مراجعة الأمان", section)

    def test_ignores_vague_or_unsubstantiated_warnings(self):
        blocks = [{"index": 1, "code": "subprocess.run([\"git\", \"status\"], check=True)\n"}]
        issues = [
            {"block": 1, "severity": "warning", "description": "Subprocess could be dangerous", "suggested_fix": "Remove subprocess."},
            {"block": 1, "severity": "warning", "evidence": "subprocess.run([\"wrong\"])" ,
             "impact": "This may cause a serious issue", "confidence": 0.99,
             "description": "Potential issue from a guessed line", "suggested_fix": "Change it."},
        ]
        self.assertEqual(normalize_review_issues(issues, blocks), [])

    def test_accepts_only_concrete_high_confidence_findings(self):
        blocks = [{"index": 1, "code": "password = request.args.get('password')\nlogin(password)\n"}]
        issues = [{
            "block": 1,
            "severity": "error",
            "evidence": "password = request.args.get('password')",
            "impact": "The password is accepted from a query parameter and may be exposed in URLs and logs.",
            "confidence": 0.96,
            "description": "Sensitive credentials are accepted through a URL query parameter.",
            "suggested_fix": "Accept the credential through a protected POST body instead.",
        }]
        normalized = normalize_review_issues(issues, blocks)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["severity"], "error")

    def test_duplicate_findings_are_collapsed(self):
        blocks = [{"index": 1, "code": "open(path, 'w').write(data)\n"}]
        finding = {
            "block": 1, "severity": "warning", "evidence": "open(path, 'w')",
            "impact": "The operation overwrites an existing file when the path is reused.",
            "confidence": 0.9, "description": "File opened in overwrite mode without preserving previous contents.",
            "suggested_fix": "Use a safe temporary output path.",
        }
        normalized = normalize_review_issues([finding, dict(finding)], blocks)
        self.assertEqual(len(normalized), 1)

    def test_warning_is_advisory_and_does_not_become_blocking(self):
        warning = {"severity": "warning", "description": "Non-blocking note"}
        error = {"severity": "error", "description": "Confirmed defect"}
        blocking, advisories = split_review_findings([warning, error])
        self.assertEqual(blocking, [error])
        self.assertEqual(advisories, [warning])

    def test_review_prompt_workflow_still_keeps_security_tool_available(self):
        self.assertIn("security_review", CODING_TOOL_GUIDANCE)
        self.assertIn("إصلاحات دفاعية فقط", CODING_TOOL_GUIDANCE["security_review"])


if __name__ == "__main__":
    unittest.main()
