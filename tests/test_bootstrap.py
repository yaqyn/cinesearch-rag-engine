import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cli.bootstrap import download_dataset, main


class BootstrapTests(unittest.TestCase):
    def test_download_validates_and_publishes_atomically(self):
        payload = {"movies": [{"id": 1, "title": "Film", "description": "Story"}]}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "data/movies.json"
            with patch("cli.bootstrap.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
                download_dataset(target)
            self.assertEqual(json.loads(target.read_text()), payload)
            with patch("cli.bootstrap.urllib.request.urlopen", return_value=io.BytesIO(b'{"movies": []}')):
                with self.assertRaises(ValueError):
                    download_dataset(target)
            self.assertEqual(json.loads(target.read_text()), payload)
            self.assertEqual(list(target.parent.glob("*.download")), [])

    def test_download_rejects_blank_titles_and_duplicate_ids(self):
        invalid_payloads = [
            {"movies": [{"id": 1, "title": "  ", "description": "Story"}]},
            {"movies": [
                {"id": 1, "title": "First", "description": "Story"},
                {"id": 1, "title": "Second", "description": "Another story"},
            ]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "movies.json"
            target.write_text('{"movies": [{"id": 9, "title": "Saved", "description": "Keep"}]}')
            for payload in invalid_payloads:
                with self.subTest(payload=payload), patch(
                    "cli.bootstrap.urllib.request.urlopen",
                    return_value=io.BytesIO(json.dumps(payload).encode()),
                ):
                    with self.assertRaises(ValueError):
                        download_dataset(target)
                    self.assertEqual(json.loads(target.read_text())["movies"][0]["title"], "Saved")

    def test_existing_data_and_complete_index_are_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "existing"
            target.touch()
            with patch.multiple("cli.bootstrap", MOVIES_PATH=target, INDEX_PATH=target,
                                DOCMAP_PATH=target, TERM_FREQUENCIES_PATH=target, DOC_LENGTHS_PATH=target), patch("cli.bootstrap.download_dataset") as download:
                main()
                download.assert_not_called()
