import importlib.util
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / ".github" / "scripts" / "release.py"
SPEC = importlib.util.spec_from_file_location("auto_release_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release
SPEC.loader.exec_module(release)


class TestReleaseScript(unittest.TestCase):
    def test_reads_this_repository(self) -> None:
        self.assertEqual(
            release.read_project(release.working_tree_reader(REPO_ROOT)),
            ("transcribe-anything", "4.1.2"),
        )

    def test_version_bump_publishes(self) -> None:
        decision = release.decide(
            "push", "refs/heads/main", "main", True, "1.0.1", "1.0.0", set()
        )
        self.assertEqual((decision.should_build, decision.should_publish), (True, True))

    def test_unchanged_or_published_version_skips(self) -> None:
        unchanged = release.decide(
            "push", "refs/heads/main", "main", True, "1.0.1", "1.0.1", set()
        )
        published = release.decide(
            "push", "refs/heads/main", "main", True, "1.0.1", "1.0.0", {"file.whl"}
        )
        self.assertFalse(unchanged.should_build)
        self.assertFalse(published.should_publish)

    def test_manual_publish_checks_existing_release(self) -> None:
        decision = release.decide(
            "workflow_dispatch",
            "refs/heads/main",
            "main",
            False,
            "1.0.1",
            None,
            {"file.whl"},
        )
        self.assertFalse(decision.should_publish)

    def test_valid_distributions_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dist = Path(directory)
            wheel = dist / "my_pkg-1.0.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "my_pkg-1.0.0.dist-info/METADATA",
                    "Metadata-Version: 2.1\nName: my-pkg\nVersion: 1.0.0\n",
                )
                archive.writestr(
                    "my_pkg-1.0.0.dist-info/entry_points.txt",
                    "[console_scripts]\nx = pkg:main\n",
                )
            (dist / "my_pkg-1.0.0.tar.gz").write_bytes(b"")
            self.assertEqual(release.verify_dist(dist, "my-pkg", "1.0.0"), [])


if __name__ == "__main__":
    unittest.main()
