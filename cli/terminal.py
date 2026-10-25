"""Cinematic, keyboard-driven catalog browser. Search backends stay independent."""

import asyncio
import json
import os
from pathlib import Path
import re
import sys

THEMES = {
    "cover": {"bg": "#211820", "fg": "#f5e5cf", "accent": "#e8af62", "surface": "#35242b"},
    "light": {"bg": "#fff6e7", "fg": "#30212b", "accent": "#713522", "surface": "#eadbc7"},
    "contrast": {"bg": "#000000", "fg": "#ffffff", "accent": "#ffff00", "surface": "#202020"},
    "terminal": {},
}


def clean(value):
    """Do not interpret dataset text as terminal control sequences."""
    value = re.sub(r"(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c|$)", "", str(value))
    value = re.sub(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]", "", value)
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", value)


def settings_path():
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "cinesearch" / "terminal.json"


def load_settings():
    try:
        data = json.loads(settings_path().read_text())
        return {"theme": data.get("theme") if data.get("theme") in THEMES else "cover",
                "spacing": data.get("spacing") if data.get("spacing") in ("compact", "comfortable") else "comfortable"}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"theme": "cover", "spacing": "comfortable"}


def save_settings(settings):
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings) + "\n")


def error_message(exc):
    if isinstance(exc, FileNotFoundError):
        return "Movie data or index files are missing. Check data/movies.json; rebuild with uv run cli/keyword_search_cli.py build."
    return f"Search failed: {clean(exc)}\nCheck local data and model availability, then try again."


def create_browser(search_catalog, theme=None, *, input=None, output=None):
    from prompt_toolkit import Application
    from prompt_toolkit.application.current import get_app_session
    terminal_output = output or get_app_session().output
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, VSplit, Layout, Window, ConditionalContainer, DynamicContainer
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import TextArea, RadioList, Button, Frame

    settings = load_settings()
    if theme:
        settings["theme"] = theme
    if "NO_COLOR" in os.environ:
        settings["theme"] = "terminal"
    state = {"results": [], "busy": False, "help": False, "focus_search": False, "status": "Describe a movie, then press Enter. Keyword search runs locally."}
    query = TextArea(height=1, multiline=False, prompt="Find a movie: ", wrap_lines=False)
    modes = RadioList([("keyword", "Keyword · local words"), ("hybrid", "Hybrid · words + meaning"), ("semantic", "Semantic · meaning")])
    results = RadioList([(-1, "Search to see movies here")])
    details = TextArea(text="Your next movie starts with a scene, a theme, or a few remembered words.\n\nTry: a bear in London", read_only=True, scrollbar=True, wrap_lines=True)

    def style():
        palette = THEMES[settings["theme"]]
        if not palette:
            return Style.from_dict({"radio-selected": "reverse", "button.focused": "reverse", "frame.label": "bold"})
        bg, fg, accent, surface = (palette[k] for k in ("bg", "fg", "accent", "surface"))
        return Style.from_dict({"": f"bg:{bg} {fg}", "frame.border": accent,
            "frame.label": f"{accent} bold", "radio-selected": f"bg:{accent} {bg} bold",
            "radio-checked": "bold", "button.focused": f"bg:{accent} {bg} bold",
            "text-area": f"bg:{surface} {fg}", "title": f"{accent} bold"})

    def update_details():
        index = results.current_value
        if isinstance(index, int) and 0 <= index < len(state["results"]):
            item = state["results"][index]
            details.text = f"{clean(item['title'])}\n\n{clean(item['description'])}\n\nRelevance score: {item['score']:.4f} (compare within this search)"

    async def run_search():
        if state["busy"]:
            return
        term = query.text.strip()
        if not term:
            state["status"] = "Enter a scene, title, or theme first."
            app.layout.focus(query)
            return
        state["focus_search"] = False
        mode = modes.current_value
        state["busy"] = True
        state["status"] = "Searching… First semantic/hybrid use may download a model. F10 exits."
        app.invalidate()
        try:
            found = await asyncio.to_thread(search_catalog, term, mode, 20)
            state["results"] = found
            results.values = [(i, clean(item["title"]).replace("\n", " ")) for i, item in enumerate(found)] or [(-1, "No matches")]
            results._selected_index = 0
            results.current_value = 0 if found else -1
            if found:
                update_details()
                if not state["focus_search"]:
                    app.layout.focus(results)
            else:
                details.text = "No movies matched. Try fewer words, another description, or a different search mode."
            state["status"] = f"{len(found)} matches for {clean(term)} · {mode}. Enter reads; Esc returns to search."
        except Exception as exc:
            state["status"] = error_message(exc)
        finally:
            state["busy"] = False
            app.invalidate()

    def start_search():
        app.create_background_task(run_search())

    def cycle_theme():
        names = list(THEMES)
        settings["theme"] = names[(names.index(settings["theme"]) + 1) % len(names)]
        persist()
        app.style = style()

    def persist():
        try:
            save_settings(settings)
            state["status"] = f"Saved: {settings['theme']} theme, {settings['spacing']} spacing."
        except OSError:
            state["status"] = "Preference applied for this session; configuration directory is not writable."

    def toggle_spacing():
        settings["spacing"] = "compact" if settings["spacing"] == "comfortable" else "comfortable"
        persist()

    bindings = KeyBindings()
    @bindings.add("tab")
    def next_focus(event):
        event.app.layout.focus_next()
    @bindings.add("s-tab")
    def previous_focus(event):
        event.app.layout.focus_previous()
    @bindings.add("c-q")
    @bindings.add("f10")
    @bindings.add("c-c")
    def quit_app(event):
        event.app.exit()
    @bindings.add("escape", eager=True)
    @bindings.add("c-f")
    def focus_query(event):
        state["help"] = False
        state["focus_search"] = True
        event.app.layout.focus(query)
    @bindings.add("f1")
    def help_toggle(event):
        state["help"] = not state["help"]
    @bindings.add("f2")
    def theme_toggle(event):
        cycle_theme()
    @bindings.add("f3")
    def spacing_toggle(event):
        toggle_spacing()
    @bindings.add("enter", filter=Condition(lambda: app.layout.has_focus(query)))
    def submit(event):
        start_search()

    in_results = Condition(lambda: app.layout.has_focus(results))
    in_modes = Condition(lambda: app.layout.has_focus(modes))

    def select_result(index):
        results._selected_index = max(0, min(index, len(results.values) - 1))
        results.current_value = results.values[results._selected_index][0]
        update_details()

    @bindings.add("up", filter=in_results, eager=True)
    @bindings.add("down", filter=in_results, eager=True)
    @bindings.add("pageup", filter=in_results, eager=True)
    @bindings.add("pagedown", filter=in_results, eager=True)
    @bindings.add("home", filter=in_results, eager=True)
    @bindings.add("end", filter=in_results, eager=True)
    def move_result(event):
        key = event.key_sequence[-1].key
        steps = {"up": -1, "down": 1, "pageup": -5, "pagedown": 5}
        index = 0 if key == "home" else len(results.values) - 1 if key == "end" else results._selected_index + steps[key]
        select_result(index)

    @bindings.add("up", filter=in_modes, eager=True)
    @bindings.add("down", filter=in_modes, eager=True)
    def move_mode(event):
        step = -1 if event.key_sequence[-1].key == "up" else 1
        modes._selected_index = (modes._selected_index + step) % len(modes.values)
        modes.current_value = modes.values[modes._selected_index][0]

    @bindings.add("enter", filter=in_results, eager=True)
    @bindings.add("right", filter=in_results, eager=True)
    def read_description(event):
        event.app.layout.focus(details)

    @bindings.add("left", filter=Condition(lambda: app.layout.has_focus(details)), eager=True)
    def back_to_results(event):
        event.app.layout.focus(results)

    @bindings.add("down", filter=Condition(lambda: app.layout.has_focus(query)), eager=True)
    def jump_to_results(event):
        if state["results"]:
            event.app.layout.focus(results)

    @bindings.add("c-s")
    def search_anywhere(event):
        start_search()

    help_text = ("Esc / Ctrl+F: return to search. Enter in search / Ctrl+S: search.\n"
                 "Tab / Shift+Tab: next / previous control. Up/Down: select mode or movie.\n"
                 "Results: Home/End, PageUp/Down; Enter/Right reads; Left returns from description.\n"
                 "F2: theme. F3: spacing. F10 / Ctrl+Q / Ctrl+C: exit.\n"
                 "Text size: terminal zoom. Screen readers: cinesearch browse --plain.")
    movie_frame = Frame(results, title="Movies · arrows select")
    detail_frame = Frame(details, title="Description · Tab to scroll")
    # Construct each layout once. Explicit dimensions prevent content length
    # from reallocating space and keep the footer anchored to the bottom.
    results.window.height = Dimension(min=1, preferred=6, weight=1)
    results.window.dont_extend_height = Condition(lambda: False)
    details.window.height = Dimension(min=1, weight=2)
    wide = VSplit([movie_frame, detail_frame], padding=1, height=Dimension(weight=1))
    narrow = HSplit([movie_frame, detail_frame], height=Dimension(weight=1))
    def result_layout():
        return narrow if terminal_output.get_size().columns < 85 else wide

    content = HSplit([
        Window(FormattedTextControl([("class:title", "CineSearch  /  Find your next story")]), height=1),
        ConditionalContainer(Window(height=1), Condition(lambda: settings["spacing"] == "comfortable")),
        query,
        Frame(modes, title="Search mode · arrows select", height=5),
        Button("Search movies", handler=start_search),
        HSplit([DynamicContainer(result_layout)], height=Dimension(min=6, weight=1)),
        ConditionalContainer(Frame(TextArea(text=help_text, read_only=True, height=6), title="Keyboard guide"), Condition(lambda: state["help"])),
        Window(FormattedTextControl(lambda: clean(state["status"])), height=3, wrap_lines=True),
        Window(FormattedTextControl("Esc Search   Enter Read   Tab Next   F1 Help   F2 Theme   F10 Exit"), height=1),
    ])
    app = Application(layout=Layout(content, focused_element=query), key_bindings=bindings,
                      style=style(), full_screen=True, mouse_support=False, input=input, output=output)
    # Escape is also the prefix of terminal key sequences. Keep the ambiguity
    # timeout short so a lone Escape feels immediate without breaking arrows.
    app.ttimeoutlen = 0.05
    app.timeoutlen = 0.3
    app.cinesearch = {"query": query, "results": results, "details": details,
                      "modes": modes, "state": state}
    return app


def launch(search_catalog, theme=None):
    from prompt_toolkit.patch_stdout import patch_stdout
    with patch_stdout():
        create_browser(search_catalog, theme).run()


def plain_browser(search_catalog):
    print("CineSearch · Plain movie search\nType /quit to exit. Results include full descriptions.")
    while True:
        try:
            query = input("Find a movie: ").strip()
            if query == "/quit":
                return
            if not query:
                continue
            found = search_catalog(query, "keyword", 10)
            print(f"{len(found)} matches")
            for i, item in enumerate(found, 1):
                print(f"\n{i}. {clean(item['title'])}\n{clean(item['description'])}")
        except (EOFError, KeyboardInterrupt):
            return
        except Exception as exc:
            print(error_message(exc), file=sys.stderr)
