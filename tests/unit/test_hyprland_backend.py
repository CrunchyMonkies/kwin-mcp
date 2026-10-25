"""The Hyprland backend without a Hyprland: IPC payload mapping, detection, layout, wrapper.

Hyprland's IPC is faked at ``hyprland.request`` with replies captured from
Hyprland 0.56.2 (``hyprctl -j clients`` / ``monitors`` / ``workspaces``), so
every test checks the KWin-shaped payloads the engine consumes and the exact
dispatcher text sent back.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pytest

from kwin_mcp import compositor, geometry, hyprland, screenshot
from kwin_mcp.compositor import CompositorTarget
from kwin_mcp.screenshot import OutputGeometry, WorkspaceTopology

if TYPE_CHECKING:
    from pathlib import Path

TARGET = CompositorTarget(name="hyprland", hyprland_signature="sig", runtime_dir="/run/user/1")

FOOT: dict[str, Any] = {
    "address": "0x55afbf70a2c0",
    "mapped": True,
    "at": [21, 21],
    "size": [1238, 678],
    "workspace": {"id": 1, "name": "1"},
    "floating": False,
    "class": "foot",
    "title": "shell",
    "pid": 4242,
    "pinned": False,
}
DIALOG: dict[str, Any] = {
    **FOOT,
    "address": "0x55afc001bf40",
    "at": [490, 274],
    "size": [300, 252],
    "workspace": {"id": 3, "name": "3"},
    "floating": True,
    "class": "zenity",
    "title": "Probe",
    "pid": 4343,
}
UNMAPPED: dict[str, Any] = {**FOOT, "address": "0x1", "mapped": False}


class FakeHyprland:
    """Answers IPC requests from mutable state and records every command."""

    def __init__(self, *, lua: bool = True) -> None:
        self.lua = lua
        self.clients: list[dict[str, Any]] = [dict(FOOT), dict(DIALOG), dict(UNMAPPED)]
        self.active = FOOT["address"]
        self.monitors: list[dict[str, Any]] = [
            {"name": "DP-1", "x": 0, "y": 0, "width": 3840, "height": 2160, "scale": 2.0},
            {"name": "HDMI-A-1", "x": 1920, "y": 0, "width": 1080, "height": 1920, "scale": 1.0}
            | {"transform": 1},
        ]
        self.workspaces = [{"id": 3, "name": "3"}, {"id": 1, "name": "1"}]
        self.current = 1
        # A fixed-size client refuses every resize.
        self.fixed: set[str] = {DIALOG["address"]}
        self.commands: list[str] = []

    def _client(self, address: str) -> dict[str, Any]:
        return next(c for c in self.clients if c["address"] == address)

    def request(self, command: str, target: CompositorTarget | None = None) -> str:
        self.commands.append(command)
        if command == "j/clients":
            return json.dumps(self.clients)
        if command == "j/activewindow":
            return json.dumps(self._client(self.active) if self.active else {})
        if command == "j/monitors":
            return json.dumps(self.monitors)
        if command == "j/workspaces":
            return json.dumps(self.workspaces)
        if command == "j/activeworkspace":
            return json.dumps({"id": self.current, "name": str(self.current)})
        if command == "dispatch hl.dsp.no_op()":
            return "ok" if self.lua else "Invalid dispatcher"
        if command.startswith("dispatch "):
            self._apply(command.removeprefix("dispatch "))
            return "ok"
        return "unknown request"

    def _apply(self, dispatcher: str) -> None:
        # Only the effects the tests observe: resizes of non-fixed clients.
        for client in self.clients:
            address = client["address"]
            if address not in dispatcher or address in self.fixed:
                continue
            if "resize" in dispatcher:
                if self.lua:
                    numbers = dispatcher.split("x = ")[1].split(", y = ")
                    client["size"] = [int(numbers[0]), int(numbers[1].split(",")[0])]
                else:
                    parts = dispatcher.split()
                    client["size"] = [int(parts[2]), int(parts[3].split(",")[0])]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeHyprland:
    state = FakeHyprland()
    monkeypatch.setattr(hyprland, "request", state.request)
    monkeypatch.setattr(hyprland, "_lua_dispatch", None)
    monkeypatch.setattr(hyprland, "_SETTLE_TIMEOUT_S", 0.0)
    return state


# ── Detection ────────────────────────────────────────────────────────────


def test_from_env_defaults_to_kwin() -> None:
    assert compositor.from_env({}) == compositor.KWIN
    assert compositor.from_env({"HYPRLAND_INSTANCE_SIGNATURE": "sig"}) == compositor.KWIN


def test_env_round_trips_through_a_helper_environment() -> None:
    assert compositor.from_env(TARGET.env()) == TARGET
    assert compositor.KWIN.env() == {"KWIN_MCP_COMPOSITOR": "kwin"}


def test_detect_live_prefers_the_exported_signature(tmp_path: Path) -> None:
    env = {"HYPRLAND_INSTANCE_SIGNATURE": "abc", "XDG_RUNTIME_DIR": str(tmp_path)}
    target = compositor.detect_live(env, wayland_display="wayland-1")
    assert target == CompositorTarget("hyprland", "abc", str(tmp_path))
    assert target.hyprland_socket() == tmp_path / "hypr" / "abc" / ".socket.sock"


def test_detect_live_finds_the_instance_serving_the_socket(tmp_path: Path) -> None:
    env = {"XDG_RUNTIME_DIR": str(tmp_path)}
    for signature, pid, socket_name in (
        ("stale", 2**22 + 12345, "wayland-1"),  # no such pid
        ("other", os.getpid(), "wayland-2"),
        ("mine", os.getpid(), "wayland-1"),
    ):
        lock = tmp_path / "hypr" / signature / "hyprland.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text(f"{pid}\n{socket_name}\n")
    assert compositor.detect_live(env, "wayland-1").hyprland_signature == "mine"
    assert compositor.detect_live(env, str(tmp_path / "wayland-2")).hyprland_signature == "other"
    assert compositor.detect_live(env, "wayland-9") == compositor.KWIN
    assert compositor.detect_live(env) == compositor.KWIN


def test_request_without_signature_fails_clearly() -> None:
    with pytest.raises(hyprland.HyprlandError, match="HYPRLAND_INSTANCE_SIGNATURE"):
        hyprland.request("j/clients", CompositorTarget(name="hyprland"))


# ── Windows, workspaces, outputs ─────────────────────────────────────────


def test_windows_map_clients_to_kwin_records(fake: FakeHyprland) -> None:
    windows = hyprland.windows(TARGET)
    assert [w["id"] for w in windows] == [FOOT["address"], DIALOG["address"]]
    foot = windows[0]
    assert foot["frame"] == foot["client"] == foot["buffer"] == [21, 21, 1238, 678]
    assert foot["app"] == "foot"
    assert foot["active"]
    assert foot["desktops"] == ["1"]
    assert not foot["onAllDesktops"]
    assert geometry.window_kind(foot) == "normal"
    assert not windows[1]["active"]


def test_pinned_window_is_on_all_desktops(fake: FakeHyprland) -> None:
    fake.clients[0]["pinned"] = True
    window = hyprland.windows(TARGET)[0]
    assert window["onAllDesktops"]
    assert window["desktops"] == []


def test_outputs_are_logical_and_rotated(fake: FakeHyprland) -> None:
    result = hyprland.outputs(TARGET)
    assert result["screens"] == [
        {"name": "DP-1", "scale": 2.0, "geometry": [0.0, 0.0, 1920.0, 1080.0]},
        {"name": "HDMI-A-1", "scale": 1.0, "geometry": [1920.0, 0.0, 1920.0, 1080.0]},
    ]
    assert result["virtual"] == [0.0, 0.0, 3840.0, 1080.0]


def test_desktops_are_workspaces_in_id_order(fake: FakeHyprland) -> None:
    result = hyprland.desktops(DIALOG["address"], TARGET)
    assert result["desktops"] == [
        {"id": "1", "name": "1", "current": True},
        {"id": "3", "name": "3", "current": False},
    ]
    assert result["current"] == "1"
    assert result["window"] == {"desktops": ["3"], "onAllDesktops": False}


@pytest.mark.parametrize(
    ("lua", "expected"),
    [
        (
            True,
            "dispatch hl.dsp.window.move({ workspace = 3, window = 'address:0x55afbf70a2c0', "
            "follow = false })",
        ),
        (False, "dispatch movetoworkspacesilent 3,address:0x55afbf70a2c0"),
    ],
)
def test_set_desktop_moves_silently(fake: FakeHyprland, lua: bool, expected: str) -> None:
    fake.lua = lua
    result = hyprland.set_desktop(FOOT["address"], "3", TARGET)
    assert result == {
        "windowFound": True,
        "desktopFound": True,
        "current": "1",
        "before": ["1"],
        "beforeOnAllDesktops": False,
    }
    assert geometry._valid_set_desktop(result)
    assert fake.commands[-1] == expected


def test_set_desktop_check_only_and_unknown_ids_send_nothing(fake: FakeHyprland) -> None:
    hyprland.set_desktop(FOOT["address"], "3", TARGET, check_only=True)
    # Hyprland creates a positive numeric workspace on demand; others must exist.
    assert hyprland.set_desktop(FOOT["address"], "7", TARGET, check_only=True)["desktopFound"]
    assert hyprland.set_desktop(FOOT["address"], "0", TARGET)["desktopFound"] is False
    assert hyprland.set_desktop(FOOT["address"], "-98", TARGET)["desktopFound"] is False
    assert hyprland.set_desktop(FOOT["address"], "special", TARGET)["desktopFound"] is False
    assert hyprland.set_desktop("0xdead", "3", TARGET)["windowFound"] is False
    assert hyprland.set_desktop(FOOT["address"], "3; exit", TARGET)["desktopFound"] is False
    assert not [c for c in fake.commands if c.startswith("dispatch hl.dsp.window")]


def test_activate_and_close_only_target_listed_windows(fake: FakeHyprland) -> None:
    assert hyprland.activate("0xdead", TARGET) == {"found": False}
    # Anything that is not a hex address never reaches a dispatcher.
    assert hyprland.close("0x1' }) hl.dsp.exit() --", TARGET) == {"found": False}
    assert hyprland.activate(DIALOG["address"], TARGET) == {"found": True}
    assert (
        fake.commands[-1] == f"dispatch hl.dsp.focus({{ window = 'address:{DIALOG['address']}' }})"
    )
    assert hyprland.close(DIALOG["address"], TARGET) == {
        "found": True,
        "closeable": True,
        "app": "zenity",
        "caption": "Probe",
    }


def test_set_geometry_floats_a_tiled_window_then_resizes_and_moves(fake: FakeHyprland) -> None:
    fake.lua = False
    result = hyprland.set_geometry(FOOT["address"], x=50, width=640, height=400, target=TARGET)
    assert result["before"] == [21, 21, 1238, 678]
    assert result["floated"] is True
    dispatched = [c for c in fake.commands if c.startswith("dispatch ") and "no_op" not in c]
    assert dispatched == [
        f"dispatch setfloating address:{FOOT['address']}",
        f"dispatch resizewindowpixel exact 640 400,address:{FOOT['address']}",
        # The unset y keeps its value from before the request.
        f"dispatch movewindowpixel exact 50 21,address:{FOOT['address']}",
    ]


def test_set_geometry_shrinks_the_box_back_to_a_refused_size(fake: FakeHyprland) -> None:
    hyprland.set_geometry(DIALOG["address"], x=100, y=80, width=500, height=300, target=TARGET)
    window = f"window = 'address:{DIALOG['address']}'"
    dispatched = [c for c in fake.commands if c.startswith("dispatch hl.dsp.window")]
    assert dispatched == [
        f"dispatch hl.dsp.window.resize({{ x = 500, y = 300, {window} }})",
        f"dispatch hl.dsp.window.resize({{ x = 300, y = 252, {window} }})",
        f"dispatch hl.dsp.window.move({{ x = 100, y = 80, {window} }})",
    ]


def test_set_geometry_rejects_sizes_beyond_the_workspace(fake: FakeHyprland) -> None:
    result = hyprland.set_geometry(FOOT["address"], width=4000, target=TARGET)
    assert result["rejected"] == "width"
    assert result["limit"] == 3840
    assert not [c for c in fake.commands if c.startswith("dispatch hl.dsp.window")]


def test_geometry_helper_routes_to_hyprland(
    fake: FakeHyprland, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in TARGET.env().items():
        monkeypatch.setenv(key, value)
    assert geometry._dispatch({"op": "active_class"}) == "foot"
    assert [w["id"] for w in geometry.collect_windows()] == [FOOT["address"], DIALOG["address"]]
    with pytest.raises(ValueError, match="Unknown geometry op"):
        geometry._dispatch({"op": "bogus"})


# ── Screenshots ──────────────────────────────────────────────────────────

TOPOLOGY = WorkspaceTopology(
    outputs=(
        OutputGeometry("DP-1", -1280, 0, 1280, 720, 2.0),
        OutputGeometry("HDMI-A-1", 0, 0, 1920, 1080, 1.0),
    ),
    virtual=(-1280, 0, 3200, 1080),
)


def test_grim_layout_places_outputs_at_the_greatest_scale() -> None:
    layout = screenshot._grim_layout((6400, 2160), TOPOLOGY)
    assert layout is not None
    placements, extents, source = layout
    assert [p.box for p in placements] == [(0, 0, 2560, 1440), (2560, 0, 6400, 2160)]
    assert extents == [(-1280, 0, 1280, 720), (0, 0, 1920, 1080)]
    assert "scale 2" in source


def test_grim_layout_refuses_an_unexpected_size() -> None:
    assert screenshot._grim_layout((3200, 1080), TOPOLOGY) is None


def test_topology_parses_hyprland_outputs(fake: FakeHyprland) -> None:
    topology = screenshot.query_topology("", TARGET)
    assert topology.virtual == (0, 0, 3840, 1080)
    assert [o.scale for o in topology.outputs] == [2.0, 1.0]


# ── Input ────────────────────────────────────────────────────────────────


def test_discrete_scroll_converts_120ths_to_detents(monkeypatch: pytest.MonkeyPatch) -> None:
    from kwin_mcp import wayland_input

    client = object.__new__(wayland_input.VirtualInputClient)
    sent: list[tuple[int, list[tuple[str, Any]]]] = []
    client._pointer = 1
    client._started = 0.0
    monkeypatch.setattr(
        client, "_marshal", lambda _proxy, opcode, args: sent.append((opcode, args))
    )
    monkeypatch.setattr(client, "_pointer_frame", lambda: None)
    client.pointer_scroll_discrete(0, -240)
    discrete = [args for opcode, args in sent if opcode == wayland_input._OP_PTR_AXIS_DISCRETE]
    assert discrete == [[("u", discrete[0][0][1]), ("u", 0), ("f", -30.0), ("i", -2)]]
    with pytest.raises(RuntimeError, match="Touch input is not available on Hyprland"):
        client.touch_down(1, 1)


# ── Virtual session ──────────────────────────────────────────────────────
