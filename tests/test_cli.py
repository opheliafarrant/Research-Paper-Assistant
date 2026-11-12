"""Regression checks for the standalone command-line entry point."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class CliTests(unittest.TestCase):
    def test_help_works_without_utf8_environment(self):
        env = os.environ.copy()
        env.pop("PYTHONIOENCODING", None)
        env["PYTHONUTF8"] = "0"
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-B", "-m", "research_assistant", "--help"],
            cwd=root, env=env, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        self.assertIn("论文阅读助手", result.stdout.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
