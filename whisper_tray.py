import sys
import os
import tempfile
import subprocess
import threading
import struct
import glob
import json
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
from PyQt6.QtWidgets import (QApplication, QSystemTrayIcon, QMenu,
                              QDialog, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QKeySequenceEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QKeySequence

# ── Keyboard backend: try pynput, fall back to raw /dev/input (Wayland-safe) ──
try:
    from pynput import keyboard
except Exception:
    # pynput 1.6.x only supports X11; use /dev/input directly instead.
    # Works on Wayland because --device=all in the Flatpak manifest grants
    # access to all /dev/input/event* nodes without any display server.
    _KEYCODE_MAP = {
        1: 'esc',    2: '1',    3: '2',    4: '3',    5: '4',    6: '5',
        7: '6',      8: '7',    9: '8',   10: '9',   11: '0',   12: 'minus',
        13: 'equal', 14: 'backspace', 15: 'tab', 28: 'enter', 57: 'space',
        16: 'q',  17: 'w',  18: 'e',  19: 'r',  20: 't',  21: 'y',  22: 'u',
        23: 'i',  24: 'o',  25: 'p',  30: 'a',  31: 's',  32: 'd',  33: 'f',
        34: 'g',  35: 'h',  36: 'j',  37: 'k',  38: 'l',  44: 'z',  45: 'x',
        46: 'c',  47: 'v',  48: 'b',  49: 'n',  50: 'm',
        59: 'f1',  60: 'f2',  61: 'f3',  62: 'f4',  63: 'f5',  64: 'f6',
        65: 'f7',  66: 'f8',  67: 'f9',  68: 'f10', 87: 'f11', 88: 'f12',
        29: 'ctrl_l',  97: 'ctrl_r',  42: 'shift_l', 54: 'shift_r',
        56: 'alt_l',  100: 'alt_r',  125: 'cmd_l',  126: 'cmd_r',
        103: 'up', 108: 'down', 105: 'left', 106: 'right',
        111: 'delete', 110: 'insert', 102: 'home', 107: 'end',
        104: 'page_up', 109: 'page_down',
    }

    class _EvdevKey:
        """Mimics pynput Key/KeyCode so existing on_press callbacks work unchanged."""
        def __init__(self, name):
            self.name = name
            self.char = name if len(name) == 1 else None

    class _EvdevListener:
        """Drop-in for pynput.keyboard.Listener that reads raw /dev/input events."""
        _EV_KEY = 1
        _FMT    = 'llHHI'
        _SZ     = struct.calcsize('llHHI')

        def __init__(self, on_press=None, on_release=None, **_):
            self._on_press   = on_press
            self._on_release = on_release
            self._running    = False

        def start(self):
            self._running = True
            for dev in glob.glob('/dev/input/event*'):
                threading.Thread(target=self._watch, args=(dev,),
                                 daemon=True).start()

        def stop(self):
            self._running = False

        def _watch(self, path):
            try:
                with open(path, 'rb') as f:
                    while self._running:
                        raw = f.read(self._SZ)
                        if not raw or len(raw) < self._SZ:
                            break
                        _, _, type_, code, value = struct.unpack(self._FMT, raw)
                        if type_ == self._EV_KEY and code in _KEYCODE_MAP:
                            key = _EvdevKey(_KEYCODE_MAP[code])
                            if value in (1, 2):    # press / auto-repeat
                                ret = self._on_press and self._on_press(key)
                                if ret is False:
                                    return
                            elif value == 0:       # release
                                ret = self._on_release and self._on_release(key)
                                if ret is False:
                                    return
            except (PermissionError, OSError):
                pass

    class _keyboard_module:
        Listener = _EvdevListener

    keyboard = _keyboard_module()

SAMPLE_RATE = 44100
SETTINGS_FILE = os.path.expanduser("~/.config/whispertype.json")

# ── Settings ──────────────────────────────────────────────────────────────
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {"hotkey": "f9"}

def save_settings(settings):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

# ── Icons ─────────────────────────────────────────────────────────────────
def make_icon(color, letter):
    px = QPixmap(64, 64)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(2, 2, 60, 60)
    p.setPen(QColor("white"))
    p.setFont(QFont("Arial", 28, QFont.Weight.Bold))
    p.drawText(px.rect(), Qt.AlignmentFlag.AlignCenter, letter)
    p.end()
    return QIcon(px)

ICON_LOADING      = lambda: make_icon("#888888", "…")
ICON_READY        = lambda: make_icon("#27ae60", "W")
ICON_RECORDING    = lambda: make_icon("#e74c3c", "●")
ICON_TRANSCRIBING = lambda: make_icon("#f39c12", "⚙")

# ── Hotkey capture dialog ─────────────────────────────────────────────────
class HotkeyDialog(QDialog):
    def __init__(self, current_key, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Hotkey")
        self.setFixedSize(340, 160)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.new_key = None
        self.listening = False

        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        self.info_label = QLabel(f"Current hotkey: <b>{current_key.upper()}</b>")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        self.capture_btn = QPushButton("🎹 Press to set new hotkey")
        self.capture_btn.setFixedHeight(40)
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background: #2c2c2c;
                color: white;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover { background: #3c3c3c; }
            QPushButton:pressed { background: #1a1a1a; }
        """)
        self.capture_btn.clicked.connect(self.start_listening)
        layout.addWidget(self.capture_btn)

        btn_row = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        self.save_btn.setFixedHeight(36)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: #27ae60;
                color: white;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover { background: #2ecc71; }
            QPushButton:disabled { background: #555; color: #888; }
        """)
        self.save_btn.clicked.connect(self.accept)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(36)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #c0392b;
                color: white;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover { background: #e74c3c; }
        """)
        cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)
        self.setLayout(layout)

        # Dark background
        self.setStyleSheet("QDialog { background: #1e1e1e; color: white; }")

        self.kb_listener = None

    def start_listening(self):
        self.capture_btn.setText("⏳ Waiting for keypress...")
        self.capture_btn.setEnabled(False)
        self.listening = True

        def on_press(key):
            if not self.listening:
                return False
            try:
                key_name = key.char.lower() if hasattr(key, 'char') and key.char else key.name
            except Exception:
                key_name = str(key).replace("Key.", "")
            self.new_key = key_name
            self.listening = False
            self.capture_btn.setText(f"✅  Key set: {key_name.upper()}")
            self.save_btn.setEnabled(True)
            return False  # stop listener

        self.kb_listener = keyboard.Listener(on_press=on_press)
        self.kb_listener.start()

    def closeEvent(self, event):
        self.listening = False
        if self.kb_listener:
            self.kb_listener.stop()
        super().closeEvent(event)

# ── Model loader thread ───────────────────────────────────────────────────
class ModelLoader(QThread):
    done = pyqtSignal(object)

    def run(self):
        model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
        self.done.emit(model)

# ── Signals ───────────────────────────────────────────────────────────────
class Signals(QObject):
    set_status = pyqtSignal(str)

# ── Main app ──────────────────────────────────────────────────────────────
class WhisperTray:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.model = None
        self.recording = False
        self.frames = []
        self.signals = Signals()
        self.signals.set_status.connect(self.update_status)
        self.settings = load_settings()

        # Tray
        self.tray = QSystemTrayIcon()
        self.tray.setIcon(ICON_LOADING())
        self.tray.setToolTip("WhisperType - Loading model...")

        # Menu
        self.menu = QMenu()
        self.status_action = self.menu.addAction("⏳ Loading model...")
        self.status_action.setEnabled(False)
        self.menu.addSeparator()
        hotkey_action = self.menu.addAction(f"⌨️  Hotkey: {self.settings['hotkey'].upper()}  —  Change")
        hotkey_action.triggered.connect(self.open_hotkey_dialog)
        self.hotkey_action = hotkey_action
        self.menu.addSeparator()
        quit_action = self.menu.addAction("⏹  Quit / Unload model")
        quit_action.triggered.connect(self.quit)
        self.tray.setContextMenu(self.menu)
        self.tray.show()

        # Load model
        self.loader = ModelLoader()
        self.loader.done.connect(self.on_model_ready)
        self.loader.start()

        # Init portal session in background
        threading.Thread(target=self.init_portal_session, daemon=True).start()

        # Keyboard listener
        self.start_kb_listener()

        sys.exit(self.app.exec())

    def start_kb_listener(self):
        if hasattr(self, 'kb_listener') and self.kb_listener:
            self.kb_listener.stop()

        def on_press(key):
            hotkey = self.settings["hotkey"]
            try:
                pressed = key.char.lower() if hasattr(key, 'char') and key.char else key.name
            except Exception:
                pressed = str(key).replace("Key.", "")
            if pressed == hotkey and not self.recording and self.model:
                self.recording = True
                self.frames = []
                self.signals.set_status.emit("recording")
                threading.Thread(target=self.record_audio, daemon=True).start()

        def on_release(key):
            hotkey = self.settings["hotkey"]
            try:
                pressed = key.char.lower() if hasattr(key, 'char') and key.char else key.name
            except Exception:
                pressed = str(key).replace("Key.", "")
            if pressed == hotkey and self.recording:
                self.recording = False

        self.kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.kb_listener.start()

    def open_hotkey_dialog(self):
        # Pause main listener while dialog captures key
        if self.kb_listener:
            self.kb_listener.stop()

        dialog = HotkeyDialog(self.settings["hotkey"])
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.new_key:
            self.settings["hotkey"] = dialog.new_key
            save_settings(self.settings)
            self.hotkey_action.setText(f"⌨️  Hotkey: {dialog.new_key.upper()}  —  Change")
            self.tray.showMessage(
                "Hotkey Updated",
                f"New hotkey: {dialog.new_key.upper()}",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )

        # Restart listener with new key
        self.start_kb_listener()

    def on_model_ready(self, model):
        self.model = model
        self.update_status("ready")
        self.tray.showMessage(
            "WhisperType Ready",
            f"Hold {self.settings['hotkey'].upper()} anywhere to record",
            QSystemTrayIcon.MessageIcon.Information,
            3000
        )

    def update_status(self, status):
        if status == "ready":
            self.tray.setIcon(ICON_READY())
            self.tray.setToolTip(f"WhisperType - Ready (Hold {self.settings['hotkey'].upper()})")
            self.status_action.setText(f"✅ Ready — Hold {self.settings['hotkey'].upper()} to talk")
        elif status == "recording":
            self.tray.setIcon(ICON_RECORDING())
            self.tray.setToolTip("WhisperType - Recording...")
            self.status_action.setText("🎙 Recording...")
        elif status == "transcribing":
            self.tray.setIcon(ICON_TRANSCRIBING())
            self.tray.setToolTip("WhisperType - Transcribing...")
            self.status_action.setText("⚙️ Transcribing...")

    def record_audio(self):
        chunks = []
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype='float32', device='pulse') as stream:
            while self.recording:
                chunk, _ = stream.read(1024)
                chunks.append(chunk)
        if chunks:
            self.signals.set_status.emit("transcribing")
            audio_data = np.concatenate(chunks, axis=0)
            threading.Thread(target=self.transcribe, args=(audio_data,), daemon=True).start()

    def init_portal_session(self):
        """Call once at startup to establish the RemoteDesktop session via GDBus."""
        try:
            from gi.repository import Gio, GLib
            import random, string

            self.portal_session_handle = None
            self._gio_bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)

            def call(iface, method, params):
                """Synchronous D-Bus call helper."""
                return self._gio_bus.call_sync(
                    'org.freedesktop.portal.Desktop',
                    '/org/freedesktop/portal/desktop',
                    iface,
                    method,
                    params,
                    None, Gio.DBusCallFlags.NONE, -1, None
                )

            def wait_for_response(request_path, timeout=30):
                """Block until the portal Request fires its Response signal."""
                result = {}
                evt = threading.Event()

                def on_signal(conn, sender, path, iface, name, params):
                    if name == 'Response' and path == request_path:
                        code, results = params.unpack()
                        if code == 0:
                            result['session'] = results.get('session_handle', '')
                        evt.set()

                sub = self._gio_bus.signal_subscribe(
                    None, 'org.freedesktop.portal.Request', 'Response',
                    request_path, None,
                    Gio.DBusSignalFlags.NONE, on_signal
                )
                evt.wait(timeout=timeout)
                self._gio_bus.signal_unsubscribe(sub)
                return result.get('session')

            tok  = ''.join(random.choices(string.ascii_lowercase, k=8))
            stok = ''.join(random.choices(string.ascii_lowercase, k=8))

            # Step 1: CreateSession
            ret = call('org.freedesktop.portal.RemoteDesktop', 'CreateSession',
                       GLib.Variant('(a{sv})', ({'handle_token': GLib.Variant('s', tok),
                                                 'session_handle_token': GLib.Variant('s', stok)},)))
            request_path = ret.unpack()[0]
            session_handle = wait_for_response(request_path)
            if not session_handle:
                raise Exception("CreateSession failed")
            self.portal_session_handle = session_handle

            # Step 2: SelectDevices (keyboard = 1)
            tok2 = ''.join(random.choices(string.ascii_lowercase, k=8))
            ret2 = call('org.freedesktop.portal.RemoteDesktop', 'SelectDevices',
                        GLib.Variant('(oa{sv})', (session_handle,
                                                  {'handle_token': GLib.Variant('s', tok2),
                                                   'types': GLib.Variant('u', 1)})))
            wait_for_response(ret2.unpack()[0])

            # Step 3: Start
            tok3 = ''.join(random.choices(string.ascii_lowercase, k=8))
            ret3 = call('org.freedesktop.portal.RemoteDesktop', 'Start',
                        GLib.Variant('(osa{sv})', (session_handle, '',
                                                   {'handle_token': GLib.Variant('s', tok3)})))
            wait_for_response(ret3.unpack()[0])

            print("✅ Portal session ready")

        except Exception as e:
            print(f"⚠️  Portal init failed: {e}")
            self.portal_session_handle = None

    def type_via_portal(self, text):
        try:
            from gi.repository import Gio, GLib

            if not hasattr(self, 'portal_session_handle') or not self.portal_session_handle:
                self.init_portal_session()

            if not self.portal_session_handle:
                print("Portal unavailable — text copied to clipboard, press Ctrl+V")
                return

            # Type each character via keysym using GDBus
            for char in text:
                keysym = ord(char)
                for state in (1, 0):  # key down, then key up
                    self._gio_bus.call_sync(
                        'org.freedesktop.portal.Desktop',
                        '/org/freedesktop/portal/desktop',
                        'org.freedesktop.portal.RemoteDesktop',
                        'NotifyKeyboardKeysym',
                        GLib.Variant('(oa{sv}iu)',
                                     (self.portal_session_handle, {},
                                      keysym, state)),
                        None, Gio.DBusCallFlags.NONE, -1, None
                    )

        except Exception as e:
            print(f"Portal typing error: {e}")
            print("Text copied to clipboard — press Ctrl+V")

    def transcribe(self, audio_data):
        try:
            audio_int16 = (audio_data * 32767).astype(np.int16)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            wav.write(tmp_path, SAMPLE_RATE, audio_int16)
            segments, _ = self.model.transcribe(tmp_path, beam_size=5, language="en")
            text = " ".join([s.text for s in segments]).strip()
            os.unlink(tmp_path)
            if text:
                print(f"📝 {text}")
                subprocess.run(["wl-copy", text])
                self.type_via_portal(text)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.signals.set_status.emit("ready")

    def quit(self):
        self.kb_listener.stop()
        self.model = None
        self.tray.hide()
        self.app.quit()

if __name__ == "__main__":
    WhisperTray()
