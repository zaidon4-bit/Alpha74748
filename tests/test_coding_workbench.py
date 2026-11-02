import unittest

from coding_workbench import (
    CODING_TOOL_GUIDANCE,
    build_coding_tool_prompt_section,
    parse_coding_tool_flags,
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


if __name__ == "__main__":
    unittest.main()
