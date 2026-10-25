"""Virtual and live Hyprland sessions, end to end through the engine.

Each test starts ``session_start(compositor="hyprland")``: a private Hyprland
nested in a headless host compositor. The live path is exercised by attaching
a second engine to that same private Hyprland with ``session_connect``, whose
detection must find it from its lock file alone. Zenity's entry dialog is the
probe: typed text is read back over AT-SPI2, and clicking OK prints it to the
app log.

Needs Hyprland, ``grim``, ``zenity`` and ``kwin_wayland``; the
KWin E2E images have none of the first three, so they skip as a known limit.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _suite_policy import require_or_skip

from kwin_mcp.core import AutomationEngine
from kwin_mcp.models import ElementPoint, Locator, PartialRect, Point, Wait, WindowSelector

if TYPE_CHECKING:
    from collections.abc import Iterator

    from kwin_mcp.models import UIElement

WIDTH, HEIGHT = 1280, 800
ZENITY = "zenity --entry --title=Probe --text=Name"


@pytest.fixture
def hyprland_engine(engine: AutomationEngine) -> AutomationEngine:
    """An engine running a virtual Hyprland session at WIDTH x HEIGHT."""
    tools = ("Hyprland", "grim", "zenity", "kwin_wayland")
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        require_or_skip("hyprland", f"Hyprland is not installed (missing {missing})", os.environ)
    engine.session_start(screen_width=WIDTH, screen_height=HEIGHT, compositor="hyprland")
    return engine


def _entry(engine: AutomationEngine) -> UIElement:
    found = engine.element_query(
        locator=Locator(app="zenity", role="text"), wait=Wait(timeout_ms=15000)
    )
    assert found.wait is not None and found.wait.satisfied, found.render()[:500]
    (entry,) = found.elements
    return entry


def _wait_for_log(engine: AutomationEngine, pid: int, text: str) -> list[str]:
    deadline = time.monotonic() + 10
    while True:
        lines = engine.app_log(pid, last_n_lines=0).lines
        if any(text in line for line in lines) or time.monotonic() > deadline:
            return lines
        time.sleep(0.1)


def test_virtual_hyprland_session_reports_its_backends(hyprland_engine: AutomationEngine) -> None:
    info = hyprland_engine.session_info().session
    assert info is not None
    assert info.compositor == "hyprland"
    assert info.input_backend == "virtual"
    assert info.screen is not None and (info.screen.width, info.screen.height) == (WIDTH, HEIGHT)
    assert "Compositor: Hyprland" in info.render_status()
    outputs = hyprland_engine._run_kwin_query({"op": "outputs"})
    assert outputs["ok"], outputs
    assert outputs["result"]["virtual"] == [0.0, 0.0, float(WIDTH), float(HEIGHT)]


def test_type_place_capture_and_click_in_a_hyprland_app(hyprland_engine: AutomationEngine) -> None:
    engine = hyprland_engine
    app = engine.app_launch(ZENITY)
    entry = _entry(engine)
    assert entry.rect is not None, "the entry must map onto the Hyprland window"

    engine.keyboard_type("Hello, Hyprland! ~|<>")
    assert _entry(engine).text == "Hello, Hyprland! ~|<>"

    (window,) = engine.window_list(WindowSelector(app_id="zenity")).windows
    assert window.active
    assert window.kind == "normal"
    placed = engine.window_place(WindowSelector(id=window.id), frame=PartialRect(x=100, y=80))
    assert placed.window is not None
    assert (placed.window.frame.x, placed.window.frame.y) == (100, 80), placed.render()
    assert not placed.pending

    frame = engine.screen_capture()
    assert frame.mapping is not None
    assert (frame.width, frame.height) == (WIDTH, HEIGHT)
    assert "backend grim" in frame.coordinate_space
    assert Path(frame.path).is_file()

    ok = engine.element_query(locator=Locator(app="zenity", role="button", query="OK"))
    (button,) = ok.elements
    assert button.rect is not None
    # The element moved with its window: its rect lies inside the placed frame.
    assert 100 <= button.rect.x < 100 + window.frame.width
    engine.pointer_click(ElementPoint(ref=button.ref))
    assert "Hello, Hyprland! ~|<>" in _wait_for_log(engine, app.pid, "Hello")


def test_window_moves_to_a_new_workspace(hyprland_engine: AutomationEngine) -> None:
    engine = hyprland_engine
    engine.app_launch(ZENITY)
    _entry(engine)
    (window,) = engine.window_list(WindowSelector(app_id="zenity")).windows
    placed = engine.window_place(WindowSelector(id=window.id), desktop="2")
    assert placed.applied.desktop, placed.render()
    assert placed.current_desktop_after == "1"
    assert "2" in {desktop.id for desktop in engine.desktop_list().desktops}


def test_touch_is_refused_with_a_remedy(hyprland_engine: AutomationEngine) -> None:
    with pytest.raises(RuntimeError, match="Touch input is not available on Hyprland"):
        hyprland_engine.touch_tap(Point(x=10, y=10))


@pytest.fixture
def live_engine(monkeypatch: pytest.MonkeyPatch) -> Iterator[AutomationEngine]:
    automation = AutomationEngine()
    # Detection must not lean on an inherited signature (e.g. the host's own Hyprland).
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    try:
        yield automation
    finally:
        automation.session_stop()


def test_session_connect_finds_hyprland_from_its_socket(
    hyprland_engine: AutomationEngine, live_engine: AutomationEngine
) -> None:
    virtual = hyprland_engine.session_info().session
    assert virtual is not None
    hyprland_engine.app_launch(ZENITY)
    _entry(hyprland_engine)

    info = live_engine.session_connect(
        dbus_address=virtual.dbus_address, wayland_display=virtual.wayland_socket
    )
    assert info.kind == "live"
    assert info.compositor == "hyprland"
    assert info.input_backend == "virtual"
    assert info.screen is not None and (info.screen.width, info.screen.height) == (WIDTH, HEIGHT)
    assert [w.app_id for w in live_engine.window_list().windows] == ["zenity"]
    live_engine.keyboard_type("live")
    assert _entry(live_engine).text == "live"
    live_engine.session_stop()
    # Disconnecting never stops the Hyprland it attached to.
    assert hyprland_engine.window_list().windows


def test_session_stop_leaves_no_hyprland_behind(hyprland_engine: AutomationEngine) -> None:
    session = hyprland_engine._session
    assert session is not None and session.info is not None
    signature = session.info.compositor.hyprland_signature
    instance = session.info.compositor.hyprland_socket().parent
    assert signature and instance.is_dir()
    pid = int((instance / "hyprland.lock").read_text().split()[0])
    hyprland_engine.session_stop()
    deadline = time.monotonic() + 5
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not Path(f"/proc/{pid}").exists()
    assert not instance.exists()
