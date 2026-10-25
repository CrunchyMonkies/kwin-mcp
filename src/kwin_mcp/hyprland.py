"""Window geometry, workspaces and outputs of a Hyprland session over its IPC socket.

The Hyprland counterpart of the KWin scripting half of ``geometry.py``: every
function here returns exactly the payload its KWin namesake returns, so the
engine, the screenshot topology and the AT-SPI coordinate mapper consume both
compositors alike. Windows are Hyprland clients addressed by their
``address`` (stable for the window's lifetime, never reused while it lives);
virtual desktops are Hyprland workspaces addressed by their numeric id.

Requests go straight to ``$XDG_RUNTIME_DIR/hypr/<signature>/.socket.sock``,
so no ``hyprctl`` binary is needed. Hyprland 0.56 and later run a Lua config
whose ``dispatch`` takes a Lua dispatcher expression (``hl.dsp.*``); older
releases and legacy ``hyprland.conf`` setups take the classic
``dispatcher args`` syntax. The syntax is probed once per process with a
no-op dispatcher.

Hyprland reports neither window types nor transient relationships, so every
client is a ``normal`` window, and a workspace move never carries over to
other windows.
"""

from __future__ import annotations

import json
import re
import socket
import time
from typing import TYPE_CHECKING, Any

from kwin_mcp import compositor

if TYPE_CHECKING:
    from kwin_mcp.compositor import CompositorTarget
    from kwin_mcp.geometry import KWinWindow

_IPC_TIMEOUT_S = 5.0
_MAX_REPLY_BYTES = 16 * 1024 * 1024
# A window address is the hex pointer Hyprland prints, e.g. 0x55afbf70a2c0.
_ADDRESS = re.compile(r"^0x[0-9a-fA-F]+$")
_WORKSPACE_ID = re.compile(r"^-?\d+$")


class HyprlandError(RuntimeError):
    """A Hyprland IPC request failed or returned something unusable."""


def request(command: str, target: CompositorTarget | None = None) -> str:
    """Send one IPC request and return Hyprland's whole reply.

    Hyprland answers a request and then closes the connection, so the reply
    is read until EOF (bounded in time and size).
    """
    target = target or compositor.from_env()
    if not target.hyprland_signature:
        msg = "HYPRLAND_INSTANCE_SIGNATURE is not set; no Hyprland instance to talk to"
        raise HyprlandError(msg)
    path = target.hyprland_socket()
    deadline = time.monotonic() + _IPC_TIMEOUT_S
    chunks: list[bytes] = []
    size = 0
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(_IPC_TIMEOUT_S)
            sock.connect(str(path))
            sock.sendall(command.encode())
            while True:
                sock.settimeout(max(0.01, deadline - time.monotonic()))
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > _MAX_REPLY_BYTES:
                    msg = f"Hyprland reply to {command.split()[0]!r} exceeds {_MAX_REPLY_BYTES} B"
                    raise HyprlandError(msg)
    except TimeoutError as exc:
        msg = f"Hyprland did not answer {command.split()[0]!r} within {_IPC_TIMEOUT_S:g}s"
        raise HyprlandError(msg) from exc
    except OSError as exc:
        msg = f"Cannot reach Hyprland IPC socket {path}: {exc}"
        raise HyprlandError(msg) from exc
    return b"".join(chunks).decode(errors="replace")


def query(command: str, target: CompositorTarget | None = None) -> Any:
    """Run a JSON query such as ``clients`` or ``monitors`` and decode it."""
    reply = request(f"j/{command}", target)
    try:
        return json.loads(reply)
    except json.JSONDecodeError as exc:
        msg = f"Hyprland returned malformed JSON for {command!r}: {reply[:200]!r}"
        raise HyprlandError(msg) from exc


_lua_dispatch: bool | None = None


def _uses_lua(target: CompositorTarget | None) -> bool:
    """Whether ``dispatch`` takes Lua (Hyprland 0.56+ with a Lua config)."""
    global _lua_dispatch
    if _lua_dispatch is None:
        _lua_dispatch = request("dispatch hl.dsp.no_op()", target).strip() == "ok"
    return _lua_dispatch


def _dispatch(lua: str, legacy: str, target: CompositorTarget | None = None) -> None:
    """Run one dispatcher in whichever syntax this Hyprland speaks."""
    command = lua if _uses_lua(target) else legacy
    reply = request(f"dispatch {command}", target).strip()
    if reply != "ok":
        msg = f"Hyprland refused {command!r}: {reply[:300]}"
        raise HyprlandError(msg)


def _lua_window(address: str) -> str:
    return f"window = 'address:{address}'"


# ── Windows ──────────────────────────────────────────────────────────────


def _active_address(target: CompositorTarget | None) -> str:
    active = query("activewindow", target)
    if isinstance(active, dict):
        return str(active.get("address", ""))
    return ""


def _box(client: dict[str, Any]) -> list[int]:
    x, y = (round(float(v)) for v in client["at"])
    width, height = (round(float(v)) for v in client["size"])
    return [x, y, width, height]


def _window_record(client: dict[str, Any], active: str) -> KWinWindow:
    box = _box(client)
    workspace = client.get("workspace") or {}
    pinned = bool(client.get("pinned", False))
    return {
        "pid": int(client.get("pid", 0)),
        "id": str(client["address"]),
        "app": str(client.get("class", "")),
        "caption": str(client.get("title", "")),
        "normal": True,
        "dialog": False,
        "utility": False,
        "popup": False,
        "managed": True,
        "deleted": False,
        "desktop": False,
        "dock": False,
        "notification": False,
        "active": str(client["address"]) == active,
        "closeable": True,
        "moveable": True,
        "resizeable": True,
        "desktops": [] if pinned else [str(workspace.get("id", ""))],
        "onAllDesktops": pinned,
        "transient": False,
        "transientFor": None,
        "modal": False,
        "frame": box,
        "client": list(box),
        "buffer": list(box),
    }


def collect_windows(target: CompositorTarget | None = None) -> list[KWinWindow]:
    """Every mapped Hyprland client as a KWin-shaped window record, in Hyprland's order."""
    clients = query("clients", target)
    if not isinstance(clients, list):
        msg = f"Hyprland returned a malformed client list: {clients!r}"
        raise HyprlandError(msg)
    active = _active_address(target)
    windows: list[KWinWindow] = []
    for client in clients:
        if not isinstance(client, dict) or not client.get("mapped", True):
            continue
        try:
            windows.append(_window_record(client, active))
        except (KeyError, TypeError, ValueError):
            continue
    return windows


def _find(window_id: str, target: CompositorTarget | None) -> KWinWindow | None:
    if not _ADDRESS.match(window_id):
        return None
    for window in collect_windows(target):
        if window["id"] == window_id:
            return window
    return None


def windows(target: CompositorTarget | None = None) -> list[KWinWindow]:
    """Every Hyprland window (all clients are listed as ``normal``)."""
    return collect_windows(target)


def activate(window_id: str, target: CompositorTarget | None = None) -> dict[str, object]:
    """Focus the window with exactly this address (switching to its workspace)."""
    if _find(window_id, target) is None:
        return {"found": False}
    _dispatch(
        f"hl.dsp.focus({{ {_lua_window(window_id)} }})",
        f"focuswindow address:{window_id}",
        target,
    )
    return {"found": True}


def set_geometry(
    window_id: str,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    target: CompositorTarget | None = None,
) -> dict[str, object]:
    """Set selected frame fields of one window, floating it first if it is tiled.

    A tiled window's geometry belongs to the layout, so the window is made
    floating before it is resized and moved. The size goes first because
    Hyprland re-centres a window it resizes into floating.
    """
    window = _find(window_id, target)
    if window is None:
        return {"found": False}
    before = list(window["frame"])
    result: dict[str, object] = {
        "found": True,
        "app": window["app"],
        "caption": window["caption"],
        "moveable": True,
        "resizeable": True,
        "before": before,
    }
    area = _workspace_area(target)
    if width is not None and area is not None and width > area[2]:
        return {**result, "rejected": "width", "limit": area[2]}
    if height is not None and area is not None and height > area[3]:
        return {**result, "rejected": "height", "limit": area[3]}

    clients = query("clients", target)
    floating = any(
        isinstance(c, dict) and c.get("address") == window_id and c.get("floating") for c in clients
    )
    if not floating:
        _dispatch(
            f"hl.dsp.window.float({{ action = 'enable', {_lua_window(window_id)} }})",
            f"setfloating address:{window_id}",
            target,
        )
        result["floated"] = True
    if width is not None or height is not None:
        w = before[2] if width is None else width
        h = before[3] if height is None else height
        _resize(window_id, w, h, target)
        # A client with a fixed or bounded size (a GTK dialog) keeps its own
        # size, but Hyprland keeps the requested box and centres the window
        # in it, which would offset the move below. Shrink the box back to
        # the size the client settled on.
        settled = _settled_size(window_id, (w, h), target)
        if settled is not None and settled != (w, h):
            _resize(window_id, settled[0], settled[1], target)
    # Always move last: floating and resizing re-centre the window, and an
    # unset coordinate must keep the value it had before the request.
    nx = before[0] if x is None else x
    ny = before[1] if y is None else y
    _dispatch(
        f"hl.dsp.window.move({{ x = {nx}, y = {ny}, {_lua_window(window_id)} }})",
        f"movewindowpixel exact {nx} {ny},address:{window_id}",
        target,
    )
    return result


_SETTLE_TIMEOUT_S = 0.5


def _resize(window_id: str, width: int, height: int, target: CompositorTarget | None) -> None:
    _dispatch(
        f"hl.dsp.window.resize({{ x = {width}, y = {height}, {_lua_window(window_id)} }})",
        f"resizewindowpixel exact {width} {height},address:{window_id}",
        target,
    )


def _settled_size(
    window_id: str, requested: tuple[int, int], target: CompositorTarget | None
) -> tuple[int, int] | None:
    """The window's size once it reached ``requested`` or stopped short of it.

    Returns None when the window is gone. Waits at most ``_SETTLE_TIMEOUT_S``
    for the client to commit a resize.
    """
    deadline = time.monotonic() + _SETTLE_TIMEOUT_S
    size: tuple[int, int] | None = None
    while True:
        window = _find(window_id, target)
        if window is None:
            return None
        size = (window["frame"][2], window["frame"][3])
        if size == requested or time.monotonic() >= deadline:
            return size
        time.sleep(0.05)


def close(window_id: str, target: CompositorTarget | None = None) -> dict[str, object]:
    """Ask the window with exactly this address to close (the app may refuse)."""
    window = _find(window_id, target)
    if window is None:
        return {"found": False}
    _dispatch(
        f"hl.dsp.window.close({{ {_lua_window(window_id)} }})",
        f"closewindow address:{window_id}",
        target,
    )
    return {"found": True, "closeable": True, "app": window["app"], "caption": window["caption"]}


def active_class(target: CompositorTarget | None = None) -> str:
    """The active window's class (the Wayland app_id), or "" when none is active."""
    active = query("activewindow", target)
    if isinstance(active, dict):
        return str(active.get("class", ""))
    return ""


# ── Workspaces ───────────────────────────────────────────────────────────


def _current_workspace(target: CompositorTarget | None) -> str:
    current = query("activeworkspace", target)
    if isinstance(current, dict) and "id" in current:
        return str(current["id"])
    return ""


def desktops(window_id: str = "", target: CompositorTarget | None = None) -> dict[str, object]:
    """Hyprland workspaces as virtual desktops, in id order, plus one window's workspace."""
    listed = query("workspaces", target)
    if not isinstance(listed, list):
        msg = f"Hyprland returned a malformed workspace list: {listed!r}"
        raise HyprlandError(msg)
    current = _current_workspace(target)
    entries = sorted(
        (w for w in listed if isinstance(w, dict) and "id" in w), key=lambda w: int(w["id"])
    )
    found: dict[str, object] | None = None
    if window_id:
        window = _find(window_id, target)
        if window is not None:
            found = {"desktops": window["desktops"], "onAllDesktops": window["onAllDesktops"]}
    return {
        "desktops": [
            {
                "id": str(w["id"]),
                "name": str(w.get("name", w["id"])),
                "current": str(w["id"]) == current,
            }
            for w in entries
        ],
        "current": current,
        "window": found,
    }


def set_desktop(
    window_id: str,
    desktop_id: str,
    target: CompositorTarget | None = None,
    *,
    check_only: bool = False,
) -> dict[str, object]:
    """Move one window to exactly one workspace without following it there.

    Accepts any workspace id Hyprland lists, and any positive workspace
    number, which Hyprland creates on demand. Hyprland has no transient
    relationships a move could carry over, so nothing is ever refused.
    """
    listed = query("workspaces", target)
    known = {str(w["id"]) for w in listed if isinstance(w, dict) and "id" in w}
    numeric = bool(_WORKSPACE_ID.match(desktop_id))
    result: dict[str, object] = {
        "windowFound": False,
        "desktopFound": numeric and (desktop_id in known or int(desktop_id) > 0),
        "current": _current_workspace(target),
    }
    window = _find(window_id, target)
    if window is None:
        return result
    result["windowFound"] = True
    result["before"] = list(window["desktops"])
    result["beforeOnAllDesktops"] = window["onAllDesktops"]
    if result["desktopFound"] and not check_only and window["desktops"] != [desktop_id]:
        _dispatch(
            f"hl.dsp.window.move({{ workspace = {desktop_id}, {_lua_window(window_id)}, "
            "follow = false })",
            f"movetoworkspacesilent {desktop_id},address:{window_id}",
            target,
        )
    return result


# ── Outputs ──────────────────────────────────────────────────────────────


def _monitor_geometry(monitor: dict[str, Any]) -> list[float]:
    """A monitor's logical rectangle: its mode divided by its scale, rotated by transform."""
    scale = float(monitor.get("scale", 1.0)) or 1.0
    width = float(monitor["width"]) / scale
    height = float(monitor["height"]) / scale
    if int(monitor.get("transform", 0)) % 2 == 1:
        width, height = height, width
    return [float(monitor["x"]), float(monitor["y"]), width, height]


def outputs(target: CompositorTarget | None = None) -> dict[str, object]:
    """Every active monitor's logical geometry and scale, plus their bounding box."""
    monitors = query("monitors", target)
    if not isinstance(monitors, list):
        msg = f"Hyprland returned a malformed monitor list: {monitors!r}"
        raise HyprlandError(msg)
    screens: list[dict[str, object]] = []
    rects: list[list[float]] = []
    for monitor in monitors:
        if not isinstance(monitor, dict) or monitor.get("disabled"):
            continue
        try:
            rect = _monitor_geometry(monitor)
            screens.append(
                {
                    "name": str(monitor["name"]),
                    "scale": float(monitor.get("scale", 1.0)),
                    "geometry": rect,
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
        rects.append(rect)
    virtual: list[float] | None = None
    if rects:
        left = min(r[0] for r in rects)
        top = min(r[1] for r in rects)
        right = max(r[0] + r[2] for r in rects)
        bottom = max(r[1] + r[3] for r in rects)
        virtual = [left, top, right - left, bottom - top]
    return {"screens": screens, "virtual": virtual}


def size_outputs(
    width: int, height: int, target: CompositorTarget | None = None, timeout: float = 5.0
) -> None:
    """Give every monitor the mode ``width``x``height`` at scale 1 and wait until it applies.

    A Hyprland nested in another compositor starts its output at the Wayland
    backend's default size and ignores the configured mode, but honours a
    monitor rule set at runtime.
    """
    monitors = query("monitors", target)
    names = [str(m["name"]) for m in monitors if isinstance(m, dict) and "name" in m]
    if not names:
        msg = "Hyprland reported no monitor to size"
        raise HyprlandError(msg)
    for name in names:
        if _uses_lua(target):
            command = (
                f'eval hl.monitor({{ output = "{name}", mode = "{width}x{height}@60", '
                'position = "0x0", scale = 1 })'
            )
        else:
            command = f"keyword monitor {name},{width}x{height}@60,0x0,1"
        reply = request(command, target).strip()
        if reply != "ok":
            msg = f"Hyprland refused the {width}x{height} mode for {name}: {reply[:200]}"
            raise HyprlandError(msg)
    deadline = time.monotonic() + timeout
    while True:
        current = [
            (round(float(m["width"])), round(float(m["height"])))
            for m in query("monitors", target)
            if isinstance(m, dict)
        ]
        if current and all(size == (width, height) for size in current):
            return
        if time.monotonic() >= deadline:
            msg = f"Hyprland outputs are {current}, not {width}x{height}, after {timeout:g}s"
            raise HyprlandError(msg)
        time.sleep(0.05)


def _workspace_area(target: CompositorTarget | None) -> list[int] | None:
    virtual = outputs(target)["virtual"]
    if not isinstance(virtual, list):
        return None
    return [round(v) for v in virtual]
