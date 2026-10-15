import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cli.cinesearch import positive_limit
from cli.terminal import clean, load_settings, save_settings, plain_browser


class TerminalTests(unittest.TestCase):
    def test_preferences_round_trip_and_corrupt_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"XDG_CONFIG_HOME": directory}):
            settings = {"theme": "contrast", "spacing": "compact"}
            save_settings(settings)
            self.assertEqual(load_settings(), settings)
            (Path(directory) / "cinesearch/terminal.json").write_text('[]')
            self.assertEqual(load_settings()["theme"], "cover")
            (Path(directory) / "cinesearch/terminal.json").write_text(json.dumps({"theme": "unknown"}))
            self.assertEqual(load_settings()["theme"], "cover")

    def test_dataset_cannot_emit_terminal_controls(self):
        self.assertNotIn("\x1b", clean("title\x1b[2J"))
        self.assertEqual(clean("فيلم\n映画"), "فيلم\n映画")

    def test_plain_mode_reads_full_description_and_exits(self):
        output = io.StringIO()
        backend = lambda *args: [{"title": "Film", "description": "Full description", "score": 1}]
        with patch("builtins.input", side_effect=["bear", "/quit"]), contextlib.redirect_stdout(output):
            plain_browser(backend)
        self.assertIn("Full description", output.getvalue())
        self.assertNotIn("\x1b", output.getvalue())

    def test_limits_reject_zero_negative_and_excessive(self):
        import argparse
        for value in ("0", "-2", "101"):
            with self.assertRaises(argparse.ArgumentTypeError):
                positive_limit(value)


class BrowserInteractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_selection_escape_and_stable_footer(self):
        import asyncio
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.data_structures import Size
        from cli.terminal import create_browser

        class Output(DummyOutput):
            columns = 110
            def get_size(self):
                return Size(rows=32, columns=self.columns)

        rows = [{"title": "Short", "description": "Short description", "score": 1},
                {"title": "Long", "description": "Long description " * 300, "score": .5}]
        with create_pipe_input() as pipe:
            output = Output()
            app = create_browser(lambda *args: rows, input=pipe, output=output)
            task = asyncio.create_task(app.run_async())
            async def settle():
                await asyncio.sleep(.15)
            try:
                await settle()
                pipe.send_text("bear\r")
                await settle()
                self.assertTrue(app.layout.has_focus(app.cinesearch["results"]))
                def footer_row():
                    screen = app.renderer.last_rendered_screen
                    return next(y for y, row in screen.data_buffer.items()
                                if "Esc Search" in "".join(row[x].char for x in sorted(row)))
                initial_row = footer_row()
                pipe.send_text("\x1b[B")
                await settle()
                self.assertEqual(app.cinesearch["results"].current_value, 1)
                self.assertIn("Long description", app.cinesearch["details"].text)
                self.assertEqual(footer_row(), initial_row)
                pipe.send_text("\r")
                await settle()
                self.assertTrue(app.layout.has_focus(app.cinesearch["details"]))
                pipe.send_text("\x1b")
                await settle()
                self.assertTrue(app.layout.has_focus(app.cinesearch["query"]))
                self.assertEqual(app.cinesearch["query"].text, "bear")
                self.assertEqual(footer_row(), initial_row)
                output.columns = 70
                app.invalidate()
                await settle()
                self.assertEqual(footer_row(), initial_row)
                pipe.send_text("\x1b[B\x1b[A")
                await settle()
                self.assertEqual(app.cinesearch["results"].current_value, 0)
                self.assertEqual(footer_row(), initial_row)
            finally:
                if app.is_running:
                    app.exit()
                await task
