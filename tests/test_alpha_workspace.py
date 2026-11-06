import io
import zipfile
import unittest

from alpha_workspace import (
    ProjectArchiveError,
    static_check_project_zip,
    static_check_generated_files,
    prepare_patched_archive,
)


def make_zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buf.getvalue()


class AlphaWorkspaceTests(unittest.TestCase):
    def test_static_checks_valid_files_without_execution(self):
        archive = make_zip({
            "app.py": "def main():\n    return 1\n",
            "config.json": '{"enabled": true}',
            "index.html": "<!doctype html><title>Alpha</title>",
            ".env": "SECRET=do-not-read\n",
        })
        result = static_check_project_zip(archive)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["passed"], 3)
        self.assertEqual(result["failed"], 0)
        self.assertGreaterEqual(result["skipped"], 1)
        self.assertIn("لم يتم تشغيل", result["notice"])

    def test_static_check_reports_syntax_errors(self):
        archive = make_zip({"broken.py": "def bad(:\n    pass\n"})
        result = static_check_project_zip(archive)
        self.assertEqual(result["failed"], 1)
        self.assertFalse(result["checks"][0]["ok"])

    def test_generated_files_receive_static_checks_without_execution(self):
        result = static_check_generated_files([
            {"path": "main.py", "content": "def main():\n    return 1\n"},
            {"path": "config.json", "content": '{"enabled": true}'},
            {"path": "broken.py", "content": "def bad(:\n    pass\n"},
        ])
        self.assertEqual(result["mode"], "static_only")
        self.assertEqual(result["passed"], 2)
        self.assertEqual(result["failed"], 1)
        self.assertIn("لم يتم تشغيل", result["notice"])

    def test_generated_file_checker_rejects_unsafe_paths(self):
        with self.assertRaises(ProjectArchiveError):
            static_check_generated_files([{"path": "../escape.py", "content": "pass"}])

    def test_generated_file_checker_rejects_secrets_and_binary_content(self):
        for path in (".env", "service-account.json", ".aws/credentials", "keys/private.pem"):
            with self.subTest(path=path):
                with self.assertRaises(ProjectArchiveError):
                    static_check_generated_files([{"path": path, "content": "secret"}])
        with self.assertRaises(ProjectArchiveError):
            static_check_generated_files([{"path": "image.png", "content": b"\x89PNG\x00binary"}])

    def test_generated_file_checker_handles_unsupported_text_and_duplicates(self):
        result = static_check_generated_files([
            {"path": "README.md", "content": "hello"},
            {"path": "app.py", "content": "print('ok')"},
        ])
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["scanned"], 1)
        with self.assertRaises(ProjectArchiveError):
            static_check_generated_files([
                {"path": "app.py", "content": "pass"},
                {"path": "app.py", "content": "pass"},
            ])

    def test_patch_preview_and_apply_only_changes_requested_existing_file(self):
        original = make_zip({"app.py": 'print("hello")\nprint("old")\nprint("bye")\n', "keep.txt": "keep\n"})
        patch = (
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,3 +1,3 @@\n"
            ' print("hello")\n'
            '-print("old")\n'
            '+print("new")\n'
            ' print("bye")\n'
        )
        updated, preview = prepare_patched_archive(original, patch)
        self.assertEqual(len(preview), 1)
        self.assertEqual(preview[0]["added"], 1)
        self.assertEqual(preview[0]["removed"], 1)
        with zipfile.ZipFile(io.BytesIO(updated)) as archive:
            self.assertEqual(archive.read("app.py").decode(), 'print("hello")\nprint("new")\nprint("bye")\n')
            self.assertEqual(archive.read("keep.txt").decode(), "keep\n")
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            self.assertEqual(archive.read("app.py").decode(), 'print("hello")\nprint("old")\nprint("bye")\n')

    def test_patch_rejects_path_traversal_and_file_creation(self):
        archive = make_zip({"app.py": "print('ok')\n"})
        traversal = "--- a/../evil.py\n+++ b/../evil.py\n@@ -1 +1 @@\n-old\n+new\n"
        creation = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+print('new')\n"
        with self.assertRaises(ProjectArchiveError):
            prepare_patched_archive(archive, traversal)
        with self.assertRaises(ProjectArchiveError):
            prepare_patched_archive(archive, creation)

    def test_patch_rejects_context_mismatch_without_mutating_source(self):
        archive = make_zip({"app.py": "print('real')\n"})
        patch = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-print('wrong')\n+print('new')\n"
        with self.assertRaises(ProjectArchiveError):
            prepare_patched_archive(archive, patch)
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            self.assertEqual(source.read("app.py").decode(), "print('real')\n")

    def test_static_checker_rejects_unsafe_zip_path(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("../outside.py", "print('bad')")
        with self.assertRaises(ProjectArchiveError):
            static_check_project_zip(buf.getvalue())

    def test_static_checker_skips_cloud_credentials_and_private_directories(self):
        archive = make_zip({
            ".aws/credentials": "aws_secret_access_key=do-not-read",
            ".ssh/id_ed25519": "private-key",
            "secrets/service-account.json": '{"private_key":"do-not-read"}',
            "src/app.py": "print('safe')",
        })
        result = static_check_project_zip(archive)
        self.assertEqual(result["scanned"], 1)
        self.assertGreaterEqual(result["skipped"], 3)
        self.assertNotIn("do-not-read", str(result["checks"]))

    def test_patch_rejects_sensitive_and_generated_paths(self):
        archive = make_zip({
            ".env": "API_KEY=hidden\n",
            "config/.api_encryption.key": "root-key\n",
            "node_modules/pkg/index.js": "module.exports = 1\n",
        })
        for path, old in [(".env", "API_KEY=hidden"), ("config/.api_encryption.key", "root-key"), ("node_modules/pkg/index.js", "module.exports = 1")]:
            patch = f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-{old}\n+changed\n"
            with self.subTest(path=path):
                with self.assertRaises(ProjectArchiveError):
                    prepare_patched_archive(archive, patch)

    def test_patch_rejects_runtime_state_files(self):
        archive = make_zip({"app.db": "db bytes\n", "app.py": "print('ok')\n"})
        patch = "--- a/app.db\n+++ b/app.db\n@@ -1 +1 @@\n-db bytes\n+changed\n"
        with self.assertRaises(ProjectArchiveError):
            prepare_patched_archive(archive, patch)


if __name__ == "__main__":
    unittest.main()
