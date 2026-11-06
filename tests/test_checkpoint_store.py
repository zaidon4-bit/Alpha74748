import unittest

from checkpoint_store import (
    CheckpointValidationError,
    checkpoint_digest,
    normalize_checkpoint_files,
    normalize_checkpoint_label,
    normalize_checkpoint_task,
)


class CheckpointStoreTests(unittest.TestCase):
    def test_normalizes_and_preserves_nested_text_files(self):
        files = normalize_checkpoint_files([
            {"path": "src/main.py", "content": "print('ok')\n", "language": "Python"},
            {"path": "README.md", "content": "# Alpha", "language": "Markdown"},
        ])
        self.assertEqual(files[0]["path"], "src/main.py")
        self.assertEqual(files[0]["content"], "print('ok')\n")
        self.assertEqual(files[1]["language"], "Markdown")

    def test_rejects_path_traversal_and_absolute_paths(self):
        for path in ("../escape.py", "/etc/passwd", "src/../../escape.py", "C:/temp/app.py", "src\\app.py"):
            with self.subTest(path=path), self.assertRaises(CheckpointValidationError):
                normalize_checkpoint_files([{"path": path, "content": "x"}])

    def test_rejects_secret_and_hidden_files_but_allows_env_example(self):
        for path in (".env", ".env.local", "secrets/key.pem", "credentials.json", ".git/config", ".aws/credentials", "private/id_rsa", "certs/signing.key"):
            with self.subTest(path=path), self.assertRaises(CheckpointValidationError):
                normalize_checkpoint_files([{"path": path, "content": "secret"}])
        files = normalize_checkpoint_files([{"path": ".env.example", "content": "API_KEY=replace-me"}])
        self.assertEqual(files[0]["path"], ".env.example")

    def test_rejects_duplicate_paths(self):
        with self.assertRaises(CheckpointValidationError):
            normalize_checkpoint_files([
                {"path": "src/app.py", "content": "one"},
                {"path": "src/app.py", "content": "two"},
            ])

    def test_rejects_non_text_and_large_files(self):
        with self.assertRaises(CheckpointValidationError):
            normalize_checkpoint_files([{"path": "app.py", "content": b"binary"}])
        with self.assertRaises(CheckpointValidationError):
            normalize_checkpoint_files([{"path": "large.txt", "content": "x" * (2 * 1024 * 1024 + 1)}])

    def test_digest_is_stable_and_changes_with_file_content(self):
        first = normalize_checkpoint_files([{"path": "a.py", "content": "print(1)"}])
        same = normalize_checkpoint_files([{"path": "a.py", "content": "print(1)"}])
        changed = normalize_checkpoint_files([{"path": "a.py", "content": "print(2)"}])
        self.assertEqual(checkpoint_digest(first), checkpoint_digest(same))
        self.assertNotEqual(checkpoint_digest(first), checkpoint_digest(changed))

    def test_label_falls_back_to_task_then_filename(self):
        files = normalize_checkpoint_files([{"path": "src/main.py", "content": "pass"}])
        self.assertEqual(normalize_checkpoint_label("", "Build a login page", files), "Build a login page")
        self.assertEqual(normalize_checkpoint_label("", "", files), "src/main.py")
        self.assertEqual(normalize_checkpoint_task(" first\n second "), "first second")


if __name__ == "__main__":
    unittest.main()
