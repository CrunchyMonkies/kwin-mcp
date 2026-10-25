"""Input injection through the wlroots-family virtual input Wayland protocols.

Hyprland has no EIS server, but it advertises ``zwlr_virtual_pointer_manager_v1``
and ``zwp_virtual_keyboard_manager_v1`` to every client. ``VirtualInputClient``
binds both on the session's Wayland socket and offers the exact method surface
of ``input.EISClient``, so ``input.InputBackend`` drives either one unchanged.

The bindings are stdlib ctypes over libwayland-client, sharing the interface
table structures of ``clipboard.py``; libxkbcommon compiles the keymap the
virtual keyboard uploads. The keymap is plain US QWERTY, the layout
``input.py``'s keycode tables assume, so typing does not depend on the
desktop's own layout: Hyprland applies the uploaded keymap to the keys of
this virtual keyboard only.

Neither protocol has a touch device, so the touch methods raise.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import os
import select
import time
from typing import TYPE_CHECKING, Any

from kwin_mcp.clipboard import _DISPATCHER_T, _Wl, _WlArgument, _WlInterface, _WlMessage

if TYPE_CHECKING:
    from collections.abc import Callable

TOUCH_UNAVAILABLE = (
    "Touch input is not available on Hyprland: its virtual input protocols "
    "offer only a pointer and a keyboard.\nUse pointer_click / pointer_drag instead."
)
DISCONNECTED = "The compositor closed the virtual input connection; input is unavailable"

_ROUNDTRIP_TIMEOUT_S = 2.0
_SYNC_TIMEOUT_S = 1.0

_OP_DISPLAY_SYNC = 0
_OP_DISPLAY_GET_REGISTRY = 1
_OP_REGISTRY_BIND = 0
_OP_VPTR_MGR_CREATE = 0
_OP_VKBD_MGR_CREATE = 0
# zwlr_virtual_pointer_v1 requests.
_OP_PTR_MOTION_ABSOLUTE = 1
_OP_PTR_BUTTON = 2
_OP_PTR_AXIS = 3
_OP_PTR_FRAME = 4
_OP_PTR_AXIS_SOURCE = 5
_OP_PTR_AXIS_STOP = 6
_OP_PTR_AXIS_DISCRETE = 7
_OP_PTR_DESTROY = 8
# zwp_virtual_keyboard_v1 requests.
_OP_KBD_KEYMAP = 0
_OP_KBD_KEY = 1
_OP_KBD_MODIFIERS = 2
_OP_KBD_DESTROY = 3

_AXIS_VERTICAL = 0
_AXIS_HORIZONTAL = 1
_AXIS_SOURCE_WHEEL = 0
_KEYMAP_FORMAT_XKB_V1 = 1
# input.py counts discrete scrolling in 120ths of a detent (libei's unit);
# axis_discrete takes whole detents plus a pixel value per detent.
_DISCRETE_UNIT = 120
_PIXELS_PER_DETENT = 15.0


# ── libxkbcommon ─────────────────────────────────────────────────────────


class _XkbRuleNames(ctypes.Structure):
    _fields_ = [
        ("rules", ctypes.c_char_p),
        ("model", ctypes.c_char_p),
        ("layout", ctypes.c_char_p),
        ("variant", ctypes.c_char_p),
        ("options", ctypes.c_char_p),
    ]


_XKB_KEY_UP = 0
_XKB_KEY_DOWN = 1
_XKB_STATE_MODS_DEPRESSED = 1 << 0
_XKB_STATE_MODS_LATCHED = 1 << 1
_XKB_STATE_MODS_LOCKED = 1 << 2
_XKB_STATE_LAYOUT_EFFECTIVE = 1 << 7
# evdev keycode + 8 is the XKB keycode.
_EVDEV_OFFSET = 8


class _UsKeymap:
    """A compiled US keymap plus the XKB state that tracks its modifiers."""

    def __init__(self) -> None:
        name = ctypes.util.find_library("xkbcommon") or "libxkbcommon.so.0"
        try:
            lib = ctypes.CDLL(name)
        except OSError as exc:
            msg = f"libxkbcommon not found ({exc}); install libxkbcommon"
            raise RuntimeError(msg) from exc
        vp = ctypes.c_void_p
        lib.xkb_context_new.restype = vp
        lib.xkb_context_new.argtypes = [ctypes.c_int]
        lib.xkb_context_unref.argtypes = [vp]
        lib.xkb_keymap_new_from_names.restype = vp
        lib.xkb_keymap_new_from_names.argtypes = [vp, ctypes.POINTER(_XkbRuleNames), ctypes.c_int]
        lib.xkb_keymap_get_as_string.restype = vp
        lib.xkb_keymap_get_as_string.argtypes = [vp, ctypes.c_int]
        lib.xkb_keymap_unref.argtypes = [vp]
        lib.xkb_state_new.restype = vp
        lib.xkb_state_new.argtypes = [vp]
        lib.xkb_state_unref.argtypes = [vp]
        lib.xkb_state_update_key.restype = ctypes.c_int
        lib.xkb_state_update_key.argtypes = [vp, ctypes.c_uint32, ctypes.c_int]
        lib.xkb_state_serialize_mods.restype = ctypes.c_uint32
        lib.xkb_state_serialize_mods.argtypes = [vp, ctypes.c_int]
        lib.xkb_state_serialize_layout.restype = ctypes.c_uint32
        lib.xkb_state_serialize_layout.argtypes = [vp, ctypes.c_int]
        libc = ctypes.CDLL(None)
        libc.free.argtypes = [vp]
        self._lib = lib
        self._context = lib.xkb_context_new(0)
        if not self._context:
            msg = "xkb_context_new failed"
            raise RuntimeError(msg)
        names = _XkbRuleNames(b"evdev", b"pc105", b"us", b"", b"")
        self._keymap = lib.xkb_keymap_new_from_names(self._context, ctypes.byref(names), 0)
        if not self._keymap:
            lib.xkb_context_unref(self._context)
            msg = "Could not compile the US XKB keymap (are the xkeyboard-config files installed?)"
            raise RuntimeError(msg)
        raw = lib.xkb_keymap_get_as_string(self._keymap, _KEYMAP_FORMAT_XKB_V1)
        try:
            # The protocol wants the keymap NUL-terminated, size included.
            self.text = ctypes.string_at(raw) + b"\0"
        finally:
            libc.free(raw)
        self._state = lib.xkb_state_new(self._keymap)

    def update(self, keycode: int, pressed: bool) -> tuple[int, int, int, int]:
        """Apply one key to the state; returns (depressed, latched, locked, group)."""
        lib = self._lib
        lib.xkb_state_update_key(
            self._state, keycode + _EVDEV_OFFSET, _XKB_KEY_DOWN if pressed else _XKB_KEY_UP
        )
        return (
            lib.xkb_state_serialize_mods(self._state, _XKB_STATE_MODS_DEPRESSED),
            lib.xkb_state_serialize_mods(self._state, _XKB_STATE_MODS_LATCHED),
            lib.xkb_state_serialize_mods(self._state, _XKB_STATE_MODS_LOCKED),
            lib.xkb_state_serialize_layout(self._state, _XKB_STATE_LAYOUT_EFFECTIVE),
        )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._lib.xkb_state_unref(self._state)
            self._lib.xkb_keymap_unref(self._keymap)
            self._lib.xkb_context_unref(self._context)


# ── Interface tables (identical to generated protocol code) ──────────────


class _Tables:
    """wl_interface definitions for the two virtual input protocols."""

    def __init__(self, wl: _Wl, output_iface: _WlInterface) -> None:
        self._keep: list[object] = []
        self.pointer = _WlInterface()
        self.pointer_manager = _WlInterface()
        self.keyboard = _WlInterface()
        self.keyboard_manager = _WlInterface()
        seat = wl.wl_seat_interface

        def types(*ifaces: _WlInterface | None) -> object:
            arr = (ctypes.POINTER(_WlInterface) * max(len(ifaces), 1))()
            for i, iface in enumerate(ifaces):
                if iface is not None:
                    arr[i] = ctypes.pointer(iface)
            self._keep.append(arr)
            return arr

        def msgs(spec: list[tuple[str, bytes, object]]) -> object:
            arr = (_WlMessage * max(len(spec), 1))()
            for i, (name, sig, ty) in enumerate(spec):
                arr[i].name = name.encode()
                arr[i].signature = sig
                arr[i].types = ty
            self._keep.append(arr)
            return arr

        def define(
            iface: _WlInterface, name: str, version: int, methods: list[tuple[str, bytes, object]]
        ) -> None:
            iface.name = name.encode()
            iface.version = version
            iface.methods = msgs(methods)
            iface.method_count = len(methods)
            iface.events = msgs([])
            iface.event_count = 0

        define(
            self.pointer,
            "zwlr_virtual_pointer_v1",
            2,
            [
                ("motion", b"uff", types(None, None, None)),
                ("motion_absolute", b"uuuuu", types(None, None, None, None, None)),
                ("button", b"uuu", types(None, None, None)),
                ("axis", b"uuf", types(None, None, None)),
                ("frame", b"", types()),
                ("axis_source", b"u", types(None)),
                ("axis_stop", b"uu", types(None, None)),
                ("axis_discrete", b"uufi", types(None, None, None, None)),
                ("destroy", b"", types()),
            ],
        )
        define(
            self.pointer_manager,
            "zwlr_virtual_pointer_manager_v1",
            2,
            [
                ("create_virtual_pointer", b"?on", types(seat, self.pointer)),
                ("destroy", b"", types()),
                (
                    "create_virtual_pointer_with_output",
                    b"2?o?on",
                    types(seat, output_iface, self.pointer),
                ),
            ],
        )
        define(
            self.keyboard,
            "zwp_virtual_keyboard_v1",
            1,
            [
                ("keymap", b"uhu", types(None, None, None)),
                ("key", b"uuu", types(None, None, None)),
                ("modifiers", b"uuuu", types(None, None, None, None)),
                ("destroy", b"", types()),
            ],
        )
        define(
            self.keyboard_manager,
            "zwp_virtual_keyboard_manager_v1",
            1,
            [("create_virtual_keyboard", b"on", types(seat, self.keyboard))],
        )


def _fixed(value: float) -> int:
    """A wl_fixed_t (24.8 signed fixed point)."""
    return round(value * 256)


class VirtualInputClient:
    """Pointer and keyboard injection with the method surface of ``input.EISClient``.

    ``wayland_display`` is the session's socket (a name under
    ``$XDG_RUNTIME_DIR`` or an absolute path). ``layout`` returns the logical
    ``(x, y, width, height)`` bounding box of the outputs: absolute pointer
    motion is a fraction of that box, so it is read before every move.
    """

    def __init__(
        self, wayland_display: str, layout: Callable[[], tuple[float, float, float, float]]
    ) -> None:
        self._layout = layout
        self._wl: _Wl | None = None
        self._display: int = 0
        self._registry: int = 0
        self._seat: int = 0
        self._pointer_manager: int = 0
        self._keyboard_manager: int = 0
        self._pointer: int = 0
        self._keyboard: int = 0
        self._keymap: _UsKeymap | None = None
        self._modifiers: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._globals: list[tuple[int, str, int]] = []
        self._handlers: dict[int, Callable[[str, list[Any]], None]] = {}
        self._disconnected = False
        self._started = time.monotonic()
        # Kept alive for as long as libwayland may call it.
        self._trampoline = _DISPATCHER_T(self._on_event)
        try:
            self._setup(wayland_display)
        except BaseException:
            self.close()
            raise

    # -- connection -----------------------------------------------------------

    def _setup(self, wayland_display: str) -> None:
        self._wl = _Wl()
        lib = self._wl.lib
        self._display = lib.wl_display_connect(wayland_display.encode() or None)
        if not self._display:
            msg = f"Cannot connect to Wayland display {wayland_display!r}"
            raise RuntimeError(msg)
        output_iface = _WlInterface.in_dll(lib, "wl_output_interface")
        tables = _Tables(self._wl, output_iface)
        self._tables = tables

        self._registry = self._marshal_new(
            self._display, _OP_DISPLAY_GET_REGISTRY, self._wl.wl_registry_interface, 1, [("n", 0)]
        )
        self._listen(self._registry, self._on_registry)
        if not self._roundtrip(_ROUNDTRIP_TIMEOUT_S):
            msg = "The compositor did not answer the Wayland registry roundtrip"
            raise RuntimeError(msg)

        def find(interface: str) -> tuple[int, int] | None:
            for name, iface, version in self._globals:
                if iface == interface:
                    return name, version
            return None

        seat = find("wl_seat")
        pointer_manager = find("zwlr_virtual_pointer_manager_v1")
        keyboard_manager = find("zwp_virtual_keyboard_manager_v1")
        missing = [
            iface
            for iface, hit in (
                ("wl_seat", seat),
                ("zwlr_virtual_pointer_manager_v1", pointer_manager),
                ("zwp_virtual_keyboard_manager_v1", keyboard_manager),
            )
            if hit is None
        ]
        if missing or seat is None or pointer_manager is None or keyboard_manager is None:
            msg = f"The compositor lacks the virtual input protocols: {', '.join(missing)}"
            raise RuntimeError(msg)

        self._seat = self._bind(seat[0], self._wl.wl_seat_interface, 1)
        self._pointer_manager = self._bind(
            pointer_manager[0], tables.pointer_manager, min(pointer_manager[1], 2)
        )
        self._keyboard_manager = self._bind(keyboard_manager[0], tables.keyboard_manager, 1)
        self._pointer = self._marshal_new(
            self._pointer_manager,
            _OP_VPTR_MGR_CREATE,
            tables.pointer,
            min(pointer_manager[1], 2),
            [("o", self._seat), ("n", 0)],
        )
        self._keyboard = self._marshal_new(
            self._keyboard_manager,
            _OP_VKBD_MGR_CREATE,
            tables.keyboard,
            1,
            [("o", self._seat), ("n", 0)],
        )
        self._keymap = _UsKeymap()
        self._upload_keymap(self._keymap.text)
        # A compositor that denies virtual input (Hyprland's permission system)
        # answers with a protocol error, which this roundtrip surfaces.
        if not self._roundtrip(_ROUNDTRIP_TIMEOUT_S) or self._display_failed():
            msg = (
                "The compositor refused the virtual pointer/keyboard "
                f"({self._display_error()}); check Hyprland's permission rules"
            )
            raise RuntimeError(msg)

    def _upload_keymap(self, text: bytes) -> None:
        fd = os.memfd_create("kwin-mcp-keymap", os.MFD_CLOEXEC)
        try:
            os.write(fd, text)
            # libwayland duplicates the fd while marshalling, so ours closes now.
            self._marshal(
                self._keyboard,
                _OP_KBD_KEYMAP,
                [("u", _KEYMAP_FORMAT_XKB_V1), ("h", fd), ("u", len(text))],
            )
            self._flush()
        finally:
            os.close(fd)

    # -- libwayland plumbing ------------------------------------------------

    def _lib(self) -> Any:
        if self._wl is None:
            msg = DISCONNECTED
            raise RuntimeError(msg)
        return self._wl.lib

    @staticmethod
    def _args(args: list[tuple[str, Any]]) -> Any:
        arr = (_WlArgument * max(len(args), 1))()
        for i, (kind, value) in enumerate(args):
            if kind == "u":
                arr[i].u = value
            elif kind in ("i", "h"):
                arr[i].i = value
            elif kind == "f":
                arr[i].f = _fixed(value)
            elif kind == "s":
                arr[i].s = value
            elif kind == "o":
                arr[i].o = value or None
            elif kind == "n":
                arr[i].n = 0
        return arr

    def _marshal(self, proxy: int, opcode: int, args: list[tuple[str, Any]]) -> None:
        self._lib().wl_proxy_marshal_array(proxy, opcode, self._args(args))

    def _marshal_new(
        self,
        proxy: int,
        opcode: int,
        iface: _WlInterface,
        version: int,
        args: list[tuple[str, Any]],
    ) -> int:
        new = self._lib().wl_proxy_marshal_array_constructor_versioned(
            proxy, opcode, self._args(args), ctypes.byref(iface), version
        )
        if not new:
            msg = f"Creating a {iface.name.decode()} proxy failed"
            raise RuntimeError(msg)
        return int(new)

    def _bind(self, name: int, iface: _WlInterface, version: int) -> int:
        arr = (_WlArgument * 4)()
        arr[0].u = name
        arr[1].s = iface.name
        arr[2].u = version
        arr[3].n = 0
        proxy = self._lib().wl_proxy_marshal_array_flags(
            self._registry, _OP_REGISTRY_BIND, ctypes.byref(iface), version, 0, arr
        )
        if not proxy:
            msg = f"Binding {iface.name.decode()} failed"
            raise RuntimeError(msg)
        return int(proxy)

    def _listen(self, proxy: int, handler: Callable[[str, list[Any]], None]) -> None:
        self._handlers[proxy] = handler
        if self._lib().wl_proxy_add_dispatcher(proxy, self._trampoline, None, None) != 0:
            msg = "wl_proxy_add_dispatcher failed"
            raise RuntimeError(msg)

    def _on_event(self, _data: int, target: int, _opcode: int, msg: Any, args: Any) -> int:
        # Never unwind through libwayland.
        with contextlib.suppress(Exception):
            name = msg.contents.name.decode()
            decoded: list[Any] = []
            signature = [c for c in msg.contents.signature.decode() if c.isalpha()]
            for i, kind in enumerate(signature):
                if kind == "s":
                    raw = args[i].s
                    decoded.append(raw.decode(errors="replace") if raw else "")
                elif kind == "u":
                    decoded.append(args[i].u)
                else:
                    decoded.append(args[i].i)
            handler = self._handlers.get(int(target))
            if handler is not None:
                handler(name, decoded)
        return 0

    def _on_registry(self, name: str, args: list[Any]) -> None:
        if name == "global":
            self._globals.append((int(args[0]), str(args[1]), int(args[2])))

    def _display_failed(self) -> bool:
        return self._display != 0 and self._lib().wl_display_get_error(self._display) != 0

    def _display_error(self) -> str:
        if not self._display:
            return "no connection"
        err = self._lib().wl_display_get_error(self._display)
        return os.strerror(err) if err else "no error"

    def _flush(self) -> None:
        if self._disconnected or not self._display:
            msg = DISCONNECTED
            raise RuntimeError(msg)
        if self._lib().wl_display_flush(self._display) < 0 or self._display_failed():
            self._disconnected = True
            msg = DISCONNECTED
            raise RuntimeError(msg)

    def _dispatch(self, timeout: float) -> None:
        """Read and dispatch whatever the compositor sent, waiting up to ``timeout``."""
        if self._disconnected or not self._display:
            return
        lib = self._lib()
        if lib.wl_display_dispatch_pending(self._display) < 0:
            self._disconnected = True
            return
        lib.wl_display_flush(self._display)
        fd = lib.wl_display_get_fd(self._display)
        readable, _, _ = select.select([fd], [], [], max(0.0, timeout))
        if readable and lib.wl_display_dispatch(self._display) < 0:
            self._disconnected = True

    def _roundtrip(self, timeout: float) -> bool:
        """A bounded wl_display.sync; False on timeout or a lost connection."""
        done: list[bool] = []
        if self._wl is None:
            return False
        callback = self._marshal_new(
            self._display, _OP_DISPLAY_SYNC, self._wl.wl_callback_interface, 1, [("n", 0)]
        )
        self._listen(callback, lambda name, _args: done.append(name == "done"))
        deadline = time.monotonic() + timeout
        try:
            while not done and not self._disconnected:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._dispatch(remaining)
            return bool(done)
        finally:
            self._handlers.pop(callback, None)
            self._lib().wl_proxy_destroy(callback)

    def _now_ms(self) -> int:
        return int((time.monotonic() - self._started) * 1000) & 0xFFFFFFFF

    # -- EISClient surface ----------------------------------------------------

    def sync(self, timeout: float = _SYNC_TIMEOUT_S) -> bool:
        """Wait until the compositor handled every request sent so far; never raises."""
        if self._disconnected or not self._display:
            return False
        try:
            return self._roundtrip(timeout)
        except (RuntimeError, OSError):
            return False

    @property
    def disconnected(self) -> bool:
        """Whether the compositor closed the connection (as of the last read)."""
        if not self._disconnected and self._display and self._display_failed():
            self._disconnected = True
        return self._disconnected

    def poll(self) -> None:
        """Handle every event already available, without waiting."""
        self._dispatch(0.0)

    def take_lost_roles(self) -> set[str]:
        """Virtual devices are never replaced by the compositor; always empty."""
        return set()

    def _pointer_frame(self) -> None:
        self._marshal(self._pointer, _OP_PTR_FRAME, [])
        self._flush()

    def pointer_move_absolute(self, x: float, y: float) -> None:
        """Move the pointer to global logical ``(x, y)``."""
        left, top, width, height = self._layout()
        extent_x, extent_y = max(1, round(width)), max(1, round(height))
        rel_x = min(max(round(x - left), 0), extent_x)
        rel_y = min(max(round(y - top), 0), extent_y)
        self._marshal(
            self._pointer,
            _OP_PTR_MOTION_ABSOLUTE,
            [("u", self._now_ms()), ("u", rel_x), ("u", rel_y), ("u", extent_x), ("u", extent_y)],
        )
        self._pointer_frame()

    def pointer_button(self, button: int, state: int) -> None:
        """Press (1) or release (0) a pointer button (evdev button code)."""
        self._marshal(
            self._pointer, _OP_PTR_BUTTON, [("u", self._now_ms()), ("u", button), ("u", state)]
        )
        self._pointer_frame()

    def pointer_scroll(self, dx: float, dy: float) -> None:
        """Scroll by a pixel delta."""
        now = self._now_ms()
        if dy:
            self._marshal(
                self._pointer, _OP_PTR_AXIS, [("u", now), ("u", _AXIS_VERTICAL), ("f", dy)]
            )
        if dx:
            self._marshal(
                self._pointer, _OP_PTR_AXIS, [("u", now), ("u", _AXIS_HORIZONTAL), ("f", dx)]
            )
        self._pointer_frame()

    def pointer_scroll_discrete(self, dx: int, dy: int) -> None:
        """Scroll by wheel detents given in 120ths (libei's convention)."""
        now = self._now_ms()
        self._marshal(self._pointer, _OP_PTR_AXIS_SOURCE, [("u", _AXIS_SOURCE_WHEEL)])
        for axis, ticks in ((_AXIS_VERTICAL, dy), (_AXIS_HORIZONTAL, dx)):
            if not ticks:
                continue
            detents = round(ticks / _DISCRETE_UNIT) or (1 if ticks > 0 else -1)
            self._marshal(
                self._pointer,
                _OP_PTR_AXIS_DISCRETE,
                [("u", now), ("u", axis), ("f", detents * _PIXELS_PER_DETENT), ("i", detents)],
            )
        self._pointer_frame()

    def pointer_scroll_stop(self) -> None:
        """End the current scroll on both axes."""
        now = self._now_ms()
        for axis in (_AXIS_VERTICAL, _AXIS_HORIZONTAL):
            self._marshal(self._pointer, _OP_PTR_AXIS_STOP, [("u", now), ("u", axis)])
        self._pointer_frame()

    def keyboard_key(self, keycode: int, state: int) -> None:
        """Press (1) or release (0) a key (evdev keycode), then publish the modifiers."""
        self._marshal(
            self._keyboard, _OP_KBD_KEY, [("u", self._now_ms()), ("u", keycode), ("u", state)]
        )
        if self._keymap is not None:
            modifiers = self._keymap.update(keycode, state != 0)
            if modifiers != self._modifiers:
                self._modifiers = modifiers
                self._marshal(self._keyboard, _OP_KBD_MODIFIERS, [("u", m) for m in modifiers])
        self._flush()

    def touch_down(self, x: float, y: float) -> int:
        raise RuntimeError(TOUCH_UNAVAILABLE)

    def touch_move(self, touch_id: int, x: float, y: float) -> None:
        raise RuntimeError(TOUCH_UNAVAILABLE)

    def touch_up(self, touch_id: int) -> None:
        raise RuntimeError(TOUCH_UNAVAILABLE)

    def close(self) -> None:
        """Destroy the virtual devices and disconnect; never raises."""
        if self._wl is not None and self._display:
            lib = self._wl.lib
            with contextlib.suppress(Exception):
                if not self._disconnected:
                    for proxy, opcode in (
                        (self._pointer, _OP_PTR_DESTROY),
                        (self._keyboard, _OP_KBD_DESTROY),
                    ):
                        if proxy:
                            lib.wl_proxy_marshal_array(proxy, opcode, self._args([]))
                    # Bounded, unlike wl_display_roundtrip: a stalled
                    # compositor must not hang session_stop.
                    self._roundtrip(_SYNC_TIMEOUT_S)
            with contextlib.suppress(Exception):
                lib.wl_display_disconnect(self._display)
        self._display = 0
        self._pointer = self._keyboard = 0
        self._handlers.clear()
        if self._keymap is not None:
            self._keymap.close()
            self._keymap = None
        self._disconnected = True
