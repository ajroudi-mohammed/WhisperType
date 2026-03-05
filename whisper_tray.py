import sys
import os
import tempfile
import subprocess
import threading
import json
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from pynput import keyboard
from faster_whisper import WhisperModel
from PyQt6.QtWidgets import (QApplication, QSystemTrayIcon, QMenu,
                              QDialog, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QKeySequenceEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QKeySequence

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
                subprocess.run(["sudo", "ydotool", "type", "--delay", "50", "--", text])
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
