import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from vespa.utils import sync_metric_presets as sync


class TestPickReleaseTag(unittest.TestCase):
    def test_skips_lsp_and_other_tags(self):
        tags = ["lsp-v2.6.0", "v8.753.16", "v8.751.13"]
        self.assertEqual("v8.753.16", sync.pick_release_tag(tags))

    def test_none_when_no_release_tag(self):
        self.assertIsNone(sync.pick_release_tag(["lsp-v2.6.0", "nightly"]))
        self.assertIsNone(sync.pick_release_tag([]))


class TestPresetDiff(unittest.TestCase):
    def test_added_and_removed(self):
        old = json.dumps(["a", "b", "c"]).encode()
        new = json.dumps(["b", "c", "d", "e"]).encode()
        self.assertEqual(
            {"added": ["d", "e"], "removed": ["a"]}, sync.preset_diff(old, new)
        )

    def test_summary_lists_changes(self):
        summary = sync.format_summary(
            "v9.0.0", "v8.753.16", {"added": ["new-one"], "removed": []}
        )
        self.assertIn("from v8.753.16 to v9.0.0", summary)
        self.assertIn("Added (1):\n- new-one", summary)
        self.assertNotIn("Removed", summary)

    def test_summary_when_only_formatting_changed(self):
        summary = sync.format_summary("v9.0.0", "v8.0.0", {"added": [], "removed": []})
        self.assertIn("No preset names changed", summary)


class TestSync(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.presets = tmp / "metric-presets.json"
        self.source = tmp / "metric-presets.source.json"
        self.presets.write_bytes(b'[\n  "a",\n  "b"\n]\n')
        self.source.write_text(
            json.dumps(
                {"repository": "vespa-engine/vespa", "path": "x.json", "ref": "v1.0.0"},
                indent=2,
            )
            + "\n"
        )
        patches = [
            mock.patch.object(sync, "PRESETS_PATH", self.presets),
            mock.patch.object(sync, "SOURCE_PATH", self.source),
            mock.patch.object(sync, "latest_release_tag", return_value="v2.0.0"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_no_change_when_identical(self):
        with mock.patch.object(
            sync, "fetch_upstream", return_value=self.presets.read_bytes()
        ):
            self.assertFalse(sync.sync())
        self.assertEqual("v1.0.0", json.loads(self.source.read_text())["ref"])

    def test_no_change_when_file_missing_at_tag(self):
        with mock.patch.object(sync, "fetch_upstream", return_value=None):
            self.assertFalse(sync.sync(tag="v0.9.0"))
        self.assertEqual(b'[\n  "a",\n  "b"\n]\n', self.presets.read_bytes())

    def test_overwrites_and_bumps_ref_on_drift(self):
        upstream = b'[\n  "a",\n  "b",\n  "c"\n]\n'
        with mock.patch.object(sync, "fetch_upstream", return_value=upstream):
            self.assertTrue(sync.sync())
        self.assertEqual(upstream, self.presets.read_bytes())
        source = json.loads(self.source.read_text())
        self.assertEqual("v2.0.0", source["ref"])
        self.assertEqual("vespa-engine/vespa", source["repository"])

    def test_dry_run_does_not_write(self):
        upstream = b'[\n  "a",\n  "b",\n  "c"\n]\n'
        with mock.patch.object(sync, "fetch_upstream", return_value=upstream):
            self.assertTrue(sync.sync(dry_run=True))
        self.assertEqual(b'[\n  "a",\n  "b"\n]\n', self.presets.read_bytes())
        self.assertEqual("v1.0.0", json.loads(self.source.read_text())["ref"])
