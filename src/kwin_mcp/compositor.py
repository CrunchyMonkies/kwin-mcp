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

import os
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
