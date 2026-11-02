import io
import zipfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from project_intelligence import (
    analyze_project_attachments,
    format_project_map_context,
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


if __name__ == "__main__":
    unittest.main()
