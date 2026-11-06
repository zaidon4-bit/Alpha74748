import io
import zipfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from project_intelligence import (
    analyze_project_attachments,
    compare_generated_project_files,
    extract_project_attachment_context,
    extract_project_source_files,
    format_project_map_context,
    normalize_ai_plan,
    redact_project_memory_text,
)


def make_zip(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class ProjectIntelligenceTests(unittest.TestCase):
    def test_maps_python_files_imports_routes_and_entrypoints(self):
        raw = make_zip({
            "app.py": (
                "from src.helpers import clean\n"
                "from flask import Flask\n"
                "app = Flask(__name__)\n"
                "@app.get('/health')\n"
                "def health():\n"
                "    return clean('ok')\n"
                "class Service:\n"
                "    pass\n"
            ),
            "src/helpers.py": "def clean(value):\n    return value.strip()\n",
            "README.md": "# Project\n",
            "requirements.txt": "flask\n",
        })
        report = analyze_project_attachments([("project.zip", "application/zip", raw)])
        self.assertEqual(report["status"], "analyzed")
        self.assertEqual(report["file_count"], 4)
        self.assertIn("app.py", report["entrypoints"])
        self.assertTrue(any(line.endswith("src/") for line in report["tree"]))
        self.assertTrue(any(edge["to"] == "src/helpers.py" for edge in report["connections"]))
        self.assertTrue(any(route["routes"] == ["GET /health"] for route in report["routes"]))
        module = next(item for item in report["python_modules"] if item["path"] == "app.py")
        self.assertIn("Service", module["classes"])
        self.assertIn("health", module["functions"])

    def test_skips_traversal_secrets_and_generated_folders(self):
        raw = make_zip({
            "../outside.py": "print('unsafe path')",
            ".env": "SECRET=do-not-echo",
            "keys/private.pem": "PRIVATE KEY",
            "node_modules/pkg/index.js": "module.exports = 1",
            "src/main.py": "def main():\n    return True\n",
        })
        report = analyze_project_attachments([("project.zip", "application/zip", raw)])
        self.assertEqual(report["status"], "analyzed")
        self.assertEqual(report["files"], ["src/main.py"])
        context = format_project_map_context(report)
        self.assertNotIn("SECRET=do-not-echo", context)
        self.assertNotIn("PRIVATE KEY", context)
        self.assertGreaterEqual(report["skipped_count"], 4)

    def test_invalid_archive_is_reported_without_extraction(self):
        report = analyze_project_attachments([("bad.zip", "application/zip", b"not a zip")])
        self.assertEqual(report["status"], "invalid_archive")
        self.assertIn("لم تُستخرج", report["summary"])
        self.assertEqual(report["files"], [])

    def test_no_files_gives_clear_inactive_result(self):
        report = analyze_project_attachments([])
        self.assertEqual(report["status"], "no_project_files")
        self.assertEqual(format_project_map_context(report), "")

    def test_standalone_source_files_are_analyzed(self):
        report = analyze_project_attachments([
            ("main.py", "text/x-python", b"def start():\n    return 1\n"),
            ("style.css", "text/css", b"body { color: black; }"),
        ])
        self.assertEqual(report["file_count"], 2)
        self.assertEqual(report["source_count"], 2)
        self.assertTrue(any(item["path"] == "main.py" for item in report["python_modules"]))

    def test_project_memory_redacts_common_credentials(self):
        source = "api_key=super-secret-value password: hunter2 Bearer eyJhbGciOi.fake-token sk-abcdefghijklmnop"
        cleaned = redact_project_memory_text(source)
        self.assertNotIn("super-secret-value", cleaned)
        self.assertNotIn("hunter2", cleaned)
        self.assertNotIn("eyJhbGciOi.fake-token", cleaned)
        self.assertNotIn("sk-abcdefghijklmnop", cleaned)
        self.assertIn("[REDACTED]", cleaned)

    def test_project_analysis_skips_cloud_and_private_credential_directories(self):
        raw = make_zip({
            ".aws/credentials": "aws_secret_access_key=do-not-send",
            ".ssh/known_hosts": "host-key-material",
            "secrets/production.json": '{"token":"do-not-send"}',
            "src/app.py": "print('safe')",
        })
        report = analyze_project_attachments([("project.zip", "application/zip", raw)])
        self.assertEqual(report["files"], ["src/app.py"])
        context = format_project_map_context(report)
        self.assertNotIn("do-not-send", context)
        self.assertGreaterEqual(report["skipped_count"], 3)

    def test_project_memory_redacts_private_key_blocks_and_aws_access_keys(self):
        source = "-----BEGIN PRIVATE KEY-----\nprivate-material\n-----END PRIVATE KEY----- AKIA1234567890ABCDEF"
        cleaned = redact_project_memory_text(source)
        self.assertNotIn("private-material", cleaned)
        self.assertNotIn("AKIA1234567890ABCDEF", cleaned)

    def test_maps_javascript_imports_and_package_dependencies(self):
        raw = make_zip({
            "src/main.js": "import { run } from './runner.js';\nexport const app = run();\n",
            "src/runner.js": "export function run() { return true; }\n",
            "package.json": '{"dependencies":{"express":"^4.0.0"},"devDependencies":{"vite":"^5.0.0"}}',
            "requirements.txt": "flask==3.0.0\nrequests>=2.0\n",
        })
        report = analyze_project_attachments([("project.zip", "application/zip", raw)])
        self.assertIn("express", report["dependencies"])
        self.assertIn("vite", report["dependencies"])
        self.assertIn("flask", report["dependencies"])
        self.assertTrue(any(item["path"] == "src/main.js" for item in report["script_modules"]))
        self.assertTrue(any(edge["from"] == "src/main.js" and edge["to"] == "src/runner.js" for edge in report["connections"]))

    def test_standalone_javascript_and_package_manifest_are_analyzed(self):
        report = analyze_project_attachments([
            ("main.js", "text/javascript", b"import helper from './helper.js';\nexport default helper;"),
            ("package.json", "application/json", b'{"dependencies":{"react":"^18"}}'),
        ])
        self.assertTrue(any(item["path"] == "main.js" for item in report["script_modules"]))
        self.assertIn("react", report["dependencies"])

    def test_generated_file_diff_labels_changed_new_and_uncompared_without_mutating_sources(self):
        original = {"src/app.py": "print('old')\n"}
        files = [
            {"path": "src/app.py", "content": "print('new')\n"},
            {"path": "src/new.py", "content": "print('new file')\n"},
            {"path": "src/unknown.py", "content": "print('unknown')\n"},
        ]
        result = compare_generated_project_files(files, original, ["src/app.py", "src/unknown.py"])
        statuses = {item["path"]: item["comparison"]["status"] for item in result}
        self.assertEqual(statuses["src/app.py"], "changed")
        self.assertEqual(statuses["src/new.py"], "new")
        self.assertEqual(statuses["src/unknown.py"], "uncompared")
        self.assertTrue(all(item["accepted"] for item in result))
        self.assertEqual(original["src/app.py"], "print('old')\n")
        self.assertNotIn("comparison", files[0])

    def test_generated_files_are_labeled_new_when_no_baseline_is_attached(self):
        result = compare_generated_project_files([
            {"path": "main.py", "content": "print('new')\n"},
        ])
        self.assertEqual(result[0]["comparison"]["status"], "new")
        self.assertGreater(result[0]["comparison"]["added"], 0)

    def test_unchanged_generated_file_has_no_diff(self):
        source = "print('same')\n"
        result = compare_generated_project_files(
            [{"path": "main.py", "content": source}], {"main.py": source}, ["main.py"]
        )
        self.assertEqual(result[0]["comparison"]["status"], "unchanged")
        self.assertEqual(result[0]["comparison"]["diff"], "")

    def test_source_file_extractor_rejects_binary_nul_bytes(self):
        sources = extract_project_source_files([
            ("safe.py", "text/x-python", b"def ok():\n    return True\n"),
            ("binary.py", "application/octet-stream", b"a\x00b"),
        ])
        self.assertEqual(list(sources), ["safe.py"])
        self.assertIn("def ok", sources["safe.py"])

    def test_uploaded_python_is_parsed_but_never_executed(self):
        with TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "must_not_exist.txt"
            source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
            raw = make_zip({"main.py": source})
            report = analyze_project_attachments([("project.zip", "application/zip", raw)])
            self.assertEqual(report["status"], "analyzed")
            self.assertFalse(marker.exists())

    def test_binary_files_are_not_listed_as_project_source(self):
        report = analyze_project_attachments([
            ("photo.png", "image/png", b"\x89PNG\r\n\x1a\n" + b"0" * 64),
        ])
        self.assertEqual(report["status"], "no_supported_files")
        self.assertEqual(report["files"], [])

    def test_prompt_context_is_bounded_and_warns_against_execution(self):
        raw = make_zip({"main.py": "def start():\n    return 1\n"})
        report = analyze_project_attachments([("project.zip", "application/zip", raw)])
        context = format_project_map_context(report)
        self.assertLessEqual(len(context), 12000)
        self.assertIn("لم تُشغّل الشيفرة", context)
        self.assertIn("بيانات غير موثوقة", context)

    def test_normalize_ai_plan_builds_valid_visual_graph(self):
        plan = normalize_ai_plan({
            "title": "نظام الطلبات", "summary": "تدفق طلب جديد", "answer": "تم.",
            "stages": [{"title": "البدء", "details": "إدخال الطلب", "deliverable": "طلب"}],
            "diagram": {"nodes": [
                {"id": "start", "label": "البداية", "type": "start"},
                {"id": "process", "label": "المعالجة", "type": "process"},
                {"id": "end", "label": "النهاية", "type": "end"}],
                "edges": [{"from": "start", "to": "process"}, {"from": "process", "to": "end"}]}
        })
        self.assertTrue(plan["structured"])
        self.assertEqual(len(plan["diagram"]["nodes"]), 3)
        self.assertEqual(len(plan["diagram"]["edges"]), 2)

    def test_normalize_ai_plan_rejects_invalid_graph_ids_and_builds_fallback(self):
        plan = normalize_ai_plan({"title": "خريطة", "stages": [{"title": "مرحلة"}],
                                  "diagram": {"nodes": [{"id": "<svg>", "label": "x"}], "edges": []}})
        self.assertGreaterEqual(len(plan["diagram"]["nodes"]), 2)
        self.assertTrue(all(node["id"].replace("_", "").isalnum() for node in plan["diagram"]["nodes"]))

    def test_extracts_bounded_source_contents_without_executing_or_including_secrets(self):
        raw = make_zip({
            "src/main.py": "def run():\n    return 'safe'\n",
            ".env": "API_KEY=never-include-this",
            "node_modules/pkg/index.js": "module.exports = true",
            "../escape.py": "print('bad')",
        })
        context, manifest = extract_project_attachment_context([("project.zip", "application/zip", raw)])
        self.assertIn("src/main.py", context)
        self.assertIn("return 'safe'", context)
        self.assertNotIn("never-include-this", context)
        self.assertNotIn("module.exports", context)
        self.assertEqual([item["path"] for item in manifest], ["src/main.py"])

    def test_extracts_standalone_source_files_and_skips_binary(self):
        context, manifest = extract_project_attachment_context([
            ("main.py", "text/x-python", b"print('ok')\n"),
            ("logo.png", "image/png", b"\x89PNG\x00binary"),
        ])
        self.assertIn("print('ok')", context)
        self.assertEqual(len(manifest), 1)

    def test_normalize_ai_plan_validates_generated_file_paths_and_content(self):
        plan = normalize_ai_plan({
            "title": "Project", "answer": "Files generated.",
            "files": [
                {"path": "src/main.py", "language": "Python", "content": "print('ok')"},
                {"path": "../outside.py", "language": "Python", "content": "unsafe"},
                {"path": ".env", "language": "Text", "content": "secret"},
            ],
        })
        self.assertEqual(len(plan["files"]), 1)
        self.assertEqual(plan["files"][0]["path"], "src/main.py")
        self.assertEqual(plan["files"][0]["content"], "print('ok')")

    def test_normalize_ai_plan_fallback_is_explicit_not_file_analysis(self):
        plan = normalize_ai_plan(None, "إجابة عادية")
        self.assertFalse(plan["structured"])
        self.assertIn("ليس تحليلاً لملفات", plan["summary"])
        self.assertEqual(plan["answer"], "إجابة عادية")


if __name__ == "__main__":
    unittest.main()
