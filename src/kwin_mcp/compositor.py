"""Which compositor a session drives: KWin or Hyprland.

Every compositor-specific backend (window geometry, output topology,
screenshots, input) picks its implementation from one ``CompositorTarget``.
The engine holds the target of its session and passes it to the in-process
backends. Helper subprocesses (the geometry helper and the AT-SPI worker)
cannot see the engine's state, so the target travels to them as environment
variables (``CompositorTarget.env``) and is read back with ``from_env``.

Kept free of D-Bus, GLib and pydantic imports: the geometry helper imports
it on every window query.
"""

from __future__ import annotations

import contextlib
import os
import re
import socket
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

CompositorName = Literal["kwin", "hyprland"]

COMPOSITOR_ENV = "KWIN_MCP_COMPOSITOR"
HYPRLAND_SIGNATURE_ENV = "HYPRLAND_INSTANCE_SIGNATURE"


@dataclass(frozen=True)
class CompositorTarget:
    """The compositor of the current session and what is needed to reach it.

    ``hyprland_signature`` names the Hyprland instance (its IPC directory
    under ``$XDG_RUNTIME_DIR/hypr``); ``runtime_dir`` is that
    ``XDG_RUNTIME_DIR``. Both are empty for KWin, which is reached over the
    session's D-Bus instead.
    """

    name: CompositorName = "kwin"
    hyprland_signature: str = ""
    runtime_dir: str = ""

    @property
    def is_hyprland(self) -> bool:
        return self.name == "hyprland"

    def env(self) -> dict[str, str]:
        """Environment variables that carry this target into a helper subprocess."""
        env = {COMPOSITOR_ENV: self.name}
        if self.is_hyprland:
            env[HYPRLAND_SIGNATURE_ENV] = self.hyprland_signature
            if self.runtime_dir:
                env["XDG_RUNTIME_DIR"] = self.runtime_dir
        return env

    def hyprland_socket(self, name: str = ".socket.sock") -> Path:
        """Path of one of this Hyprland instance's IPC sockets."""
        return hyprland_instance_dir(self.runtime_dir, self.hyprland_signature) / name


KWIN = CompositorTarget()


def hyprland_instance_dir(runtime_dir: str, signature: str) -> Path:
    """``$XDG_RUNTIME_DIR/hypr/<signature>``, where Hyprland keeps its IPC sockets."""
    base = runtime_dir or os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    path = Path(base) / "hypr"
    return path / signature if signature else path


def from_env(env: Mapping[str, str] | None = None) -> CompositorTarget:
    """The target ``env`` (default: this process's environment) carries.

    Without ``KWIN_MCP_COMPOSITOR`` the target is KWin, which keeps every
    existing caller of the helpers unchanged.
    """
    source = os.environ if env is None else env
    if source.get(COMPOSITOR_ENV) != "hyprland":
        return KWIN
    return CompositorTarget(
        name="hyprland",
        hyprland_signature=source.get(HYPRLAND_SIGNATURE_ENV, ""),
        runtime_dir=source.get("XDG_RUNTIME_DIR", ""),
    )


def _signature_for_socket(runtime_dir: str, wayland_display: str) -> str:
    """The running Hyprland instance serving ``wayland_display``, or "".

    Each instance writes ``<pid>\\n<wayland socket>`` to
    ``hypr/<signature>/hyprland.lock``; a lock whose pid is gone is stale.
    """
    socket_name = Path(wayland_display).name
    base = hyprland_instance_dir(runtime_dir, "")
    try:
        locks = sorted(base.glob("*/hyprland.lock"))
    except OSError:
        return ""
    for lock in locks:
        try:
            lines = lock.read_text().splitlines()
        except OSError:
            continue
        if len(lines) < 2 or lines[1].strip() != socket_name or not lines[0].strip().isdigit():
            continue
        if Path(f"/proc/{lines[0].strip()}").exists():
            return lock.parent.name
    return ""


def detect_live(
    env: Mapping[str, str] | None = None, wayland_display: str = ""
) -> CompositorTarget:
    """The compositor of a live desktop, judged from its session environment.

    Hyprland exports ``HYPRLAND_INSTANCE_SIGNATURE`` to every client it
    starts. A process started outside the desktop (a service, an IDE
    backend) may lack it, so the Hyprland instance whose lock file names
    ``wayland_display`` counts too. Any other desktop is treated as KWin,
    whose reachability the caller checks over D-Bus.
    """
    source = os.environ if env is None else env
    runtime_dir = source.get("XDG_RUNTIME_DIR", "")
    signature = source.get(HYPRLAND_SIGNATURE_ENV, "")
    if not signature and wayland_display:
        signature = _signature_for_socket(runtime_dir, wayland_display)
    if not signature:
        return KWIN
    return CompositorTarget(
        name="hyprland",
        hyprland_signature=signature,
        runtime_dir=runtime_dir,
    )


# ── Finding the running desktop ──────────────────────────────────────────

# kwin-mcp's own virtual Hyprland sessions run with this in their environment
# (see session._hyprland_block); they are never the user's desktop.
_VIRTUAL_HYPRLAND_MARKER = b"AQ_DRM_DEVICES=/nonexistent"
_WAYLAND_SOCKET = re.compile(r"^wayland-(\d+)$")


def runtime_dir(env: Mapping[str, str] | None = None) -> str:
    """``$XDG_RUNTIME_DIR``, or the systemd default for this user."""
    source = os.environ if env is None else env
    return source.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"


def _hyprland_desktops(base: str) -> list[tuple[str, str]]:
    """``(signature, socket)`` of every running Hyprland that is not kwin-mcp's, oldest first.

    The signature embeds the instance's start time (``<hash>_<time>_<random>``),
    so the oldest instance is the one the login session started.
    """
    found: list[tuple[int, str, str]] = []
    try:
        locks = list(hyprland_instance_dir(base, "").glob("*/hyprland.lock"))
    except OSError:
        return []
    for lock in locks:
        try:
            pid, socket_name = (line.strip() for line in lock.read_text().splitlines()[:2])
        except (OSError, ValueError):
            continue
        if not pid.isdigit() or not Path(f"/proc/{pid}").exists():
            continue
        with contextlib.suppress(OSError):
            if _VIRTUAL_HYPRLAND_MARKER in Path(f"/proc/{pid}/environ").read_bytes():
                continue
        parts = lock.parent.name.split("_")
        started = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else 0
        found.append((started, lock.parent.name, socket_name))
    return [(signature, socket_name) for _, signature, socket_name in sorted(found)]


def _accepts_connections(path: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            probe.connect(str(path))
    except OSError:
        return False
    return True


def discover_live_display(env: Mapping[str, str] | None = None) -> str:
    """The Wayland socket of the running desktop, for a server started outside it; or "".

    Used only when neither ``wayland_display`` nor ``$WAYLAND_DISPLAY`` names
    one (an MCP server launched by a service or an IDE backend). In order: the
    Hyprland named by ``$HYPRLAND_INSTANCE_SIGNATURE``; the oldest running
    Hyprland that is not a kwin-mcp virtual session; the lowest-numbered
    ``wayland-N`` socket that accepts connections (KWin and every other
    compositor). kwin-mcp's own virtual KWin sockets are named
    ``wayland-mcp-*`` and never match.
    """
    source = os.environ if env is None else env
    base = runtime_dir(source)
    signature = source.get(HYPRLAND_SIGNATURE_ENV, "")
    if signature and "/" not in signature:
        with contextlib.suppress(OSError, IndexError):
            lines = (hyprland_instance_dir(base, signature) / "hyprland.lock").read_text()
            socket_name = lines.splitlines()[1].strip()
            if socket_name:
                return socket_name
    for _, socket_name in _hyprland_desktops(base):
        if _accepts_connections(Path(base) / socket_name):
            return socket_name
    try:
        names = [p.name for p in Path(base).iterdir() if _WAYLAND_SOCKET.match(p.name)]
    except OSError:
        return ""
    for name in sorted(names, key=lambda n: int(n.split("-")[1])):
        path = Path(base) / name
        with contextlib.suppress(OSError):
            if stat.S_ISSOCK(path.stat().st_mode) and _accepts_connections(path):
                return name
    return ""


def discover_session_bus(env: Mapping[str, str] | None = None) -> str:
    """``unix:path=$XDG_RUNTIME_DIR/bus`` when that socket exists (systemd's user bus); or ""."""
    path = Path(runtime_dir(env)) / "bus"
    with contextlib.suppress(OSError):
        if stat.S_ISSOCK(path.stat().st_mode):
            return f"unix:path={path}"
    return ""
