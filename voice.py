#!/usr/bin/env python3
"""
VoiceEngine - Python menubar dictation app.

Usage:
  python3 voice.py                    # Menubar app (Caps Lock to dictate)
  python3 voice.py --file test.wav    # Transcribe a file, print result

Requires: pip install rumps pyobjc-core pyobjc-framework-CoreML
          numpy coremltools transformers sentencepiece
"""

import os, sys, struct, wave, time, json, threading
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from pathlib import Path

MODEL_DIR = Path.home() / ".cache/moonshine-coreml/tiny-streaming"
SR = 16000

# ── Core Inference ────────────────────────────────────────────────────────────

class MoonshineSTT:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.kw = self.vw = self.kb = self.vb = None
        self.cos_tables = self.sin_tables = None
        self.consts = {}
        self.tokenizer = None
        self.ready = False

    def load(self):
        import coremltools as ct
        import numpy as np
        from transformers import AutoTokenizer

        if self.ready:
            return
        import logging
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        enc_path = MODEL_DIR / "encoder.mlpackage"
        dec_path = MODEL_DIR / "decoder_stateful.mlpackage"
        w_path = MODEL_DIR / "cross_kv_weights.npz"

        self.encoder = ct.models.MLModel(str(enc_path))
        self.decoder = ct.models.MLModel(str(dec_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        self.tokenizer = AutoTokenizer.from_pretrained("UsefulSensors/moonshine-tiny")

        w = np.load(str(w_path))
        self.consts = {k: int(w[k]) for k in ["NL", "H", "D", "HID", "S_MAX"]}
        self.kw = [w[f"layer{i}_k_weight"] for i in range(self.consts["NL"])]
        self.vw = [w[f"layer{i}_v_weight"] for i in range(self.consts["NL"])]
        self.kb = [w.get(f"layer{i}_k_bias") for i in range(self.consts["NL"])]
        self.vb = [w.get(f"layer{i}_v_bias") for i in range(self.consts["NL"])]

        ct_ = w["cos_tables"]
        st_ = w["sin_tables"]
        self.cos_tables = [ct_[i].reshape(1, 1, 1, -1) for i in range(self.consts["S_MAX"])]
        self.sin_tables = [st_[i].reshape(1, 1, 1, -1) for i in range(self.consts["S_MAX"])]

        # Warm up ANE
        import numpy as np
        dummy = np.zeros((1, 160000), dtype=np.float32)
        self.encoder.predict({"audio": dummy})

        self.ready = True

    def transcribe(self, audio, return_timing=False):
        import numpy as np
        t0 = time.perf_counter()

        # Pad/clip to 160000
        if len(audio) < 160000:
            audio = np.pad(audio, (0, 160000 - len(audio)))
        else:
            audio = audio[:160000]

        # Encoder
        out = self.encoder.predict({"audio": audio.reshape(1, -1).astype(np.float32)})
        hidden = out["hidden_states"][0]
        S_enc = hidden.shape[0]

        t1 = time.perf_counter()

        # Cross-KV projection
        NL, H, D, HID = self.consts["NL"], self.consts["H"], self.consts["D"], self.consts["HID"]
        HD, S_MAX = H * D, self.consts["S_MAX"]
        S_ENC_MAX = 500

        cross_k = np.zeros((NL, 1, H, S_ENC_MAX, D), dtype=np.float32)
        cross_v = np.zeros((NL, 1, H, S_ENC_MAX, D), dtype=np.float32)
        cross_mask = np.full((1, 1, 1, S_ENC_MAX), -10000.0, dtype=np.float32)
        cross_mask[0, 0, 0, :S_enc] = 0.0

        for i in range(NL):
            k = hidden @ self.kw[i].T
            v = hidden @ self.vw[i].T
            if self.kb[i] is not None:
                k += self.kb[i].reshape(1, -1)
            if self.vb[i] is not None:
                v += self.vb[i].reshape(1, -1)
            k = k.reshape(S_enc, H, D)
            v = v.reshape(S_enc, H, D)
            # Vectorized: transpose (S_enc, H, D) -> (H, S_enc, D) 
            cross_k[i, 0, :, :S_enc, :] = k.transpose(1, 0, 2)
            cross_v[i, 0, :, :S_enc, :] = v.transpose(1, 0, 2)

        t2 = time.perf_counter()

        # Decoder loop
        state = self.decoder.make_state()
        state.write_state("cross_k", cross_k)
        state.write_state("cross_v", cross_v)
        state.write_state("cross_mask", cross_mask)

        BOS, EOS = 1, 2
        token_ids = [BOS]
        attn_mask = np.full((1, 1, 1, S_MAX), -10000.0, dtype=np.float32)
        attn_mask[0, 0, 0, 0] = 0.0

        for step in range(min(S_MAX, 200)):
            inp = {
                "input_ids": np.array([[token_ids[-1]]], dtype=np.int32),
                "attn_mask": attn_mask,
                "cos": self.cos_tables[step],
                "sin": self.sin_tables[step],
                "write_onehot": np.zeros((1, 1, S_MAX, 1), dtype=np.float32),
            }
            inp["write_onehot"][0, 0, step, 0] = 1.0

            out = self.decoder.predict(inp, state=state)
            logits = out["logits"][0, 0]
            top = int(logits.argmax())
            token_ids.append(top)
            if top == EOS:
                break

            next_pos = step + 1
            if next_pos < S_MAX:
                attn_mask[0, 0, 0, next_pos] = 0.0

        t3 = time.perf_counter()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

        if return_timing:
            return text, {"encoder": (t1-t0)*1000, "kv": (t2-t1)*1000, "decoder": (t3-t2)*1000, "total": (t3-t0)*1000}
        return text


# ── Audio Capture ─────────────────────────────────────────────────────────────

class AudioCapture:
    """Record from default mic via ffmpeg (stdout pipe)."""
    def __init__(self):
        self._proc = None
        self._data = bytearray()

    def start(self):
        import subprocess
        self._data = bytearray()
        self._proc = subprocess.Popen([
            "ffmpeg", "-loglevel", "quiet",
            "-f", "avfoundation",
            "-i", ":0",              # default input mic
            "-ar", "16000",           # resample to 16kHz
            "-ac", "1",               # mono
            "-sample_fmt", "s16",     # 16-bit signed
            "-f", "wav",
            "pipe:1"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        import threading
        def _reader():
            while self._proc and self._proc.poll() is None:
                try:
                    chunk = self._proc.stdout.read(65536)
                    if not chunk: break
                    self._data.extend(chunk)
                except:
                    break
        threading.Thread(target=_reader, daemon=True).start()

    def stop(self):
        import time
        if self._proc:
            self._proc.terminate()
            try: self._proc.wait(timeout=3)
            except: self._proc.kill()
            self._proc = None
            time.sleep(0.3)

    def take_buffer(self):
        import numpy as np
        if not self._data:
            return None
        try:
            import io
            with wave.open(io.BytesIO(bytes(self._data))) as w:
                n = w.getnframes()
                raw = w.readframes(n)
                samples = struct.unpack("<" + "h" * n, raw)
            return np.array(samples, dtype=np.float32) / 32768.0
        except Exception as e:
            print(f"WAV parse error: {e}")
            return None



# ── Caps Lock Monitor (CGEvent tap) ──────────────────────────────────────────

class CapsLockMonitor:
    """Global hotkey via CGEvent tap. Press Caps Lock to start/stop recording."""
    def __init__(self, on_press=None, on_release=None):
        self._on_press = on_press or (lambda: None)
        self._on_release = on_release or (lambda: None)
        self._tap = None

    def start(self):
        import Quartz

        mask = (1 << Quartz.kCGEventFlagsChanged)

        self._prev_caps = False

        def callback(proxy, tap_type, event, refcon):
            if event.type() == Quartz.kCGEventFlagsChanged:
                keycode = event.getIntegerValueField(Quartz.kCGKeyboardEventKeycode)
                if keycode == 57:
                    flags = event.getIntegerValueField(Quartz.kCGEventFlagMask)
                    is_down = (flags & 65536) != 0  # kCGEventFlagMaskAlphaShift
                    if is_down and not self._prev_caps:
                        self._prev_caps = True
                        self._on_press()
                    elif not is_down and self._prev_caps:
                        self._prev_caps = False
                        self._on_release()
                    return None
            return event

        self._tap = Quartz.CGEvent.tapCreate(
            tap=Quartz.kCGSessionEventTap,
            place=Quartz.kCGHeadInsertEventTap,
            options=Quartz.kCGEventTapOptionDefault,
            eventsOfInterest=mask,
            callback=callback,
            userInfo=None,
        )
        if self._tap is None:
            print("Caps Lock tap failed — grant Accessibility in System Settings > Privacy")
            return

        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Quartz.CFRunLoopAddSource(
            Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopDefaultMode
        )
        Quartz.CGEvent.tapEnable(self._tap, True)


# ── Menubar App (rumps) ──────────────────────────────────────────────────────

try:
    import rumps

    class VoiceApp(rumps.App):
        def __init__(self, stt):
            super().__init__("VoiceEngine", icon=None)
            self.stt = stt
            self.capture = None
            self.recording = False
            self.menu = ["Status: idle", None, "Quit"]

        @rumps.clicked("Status: idle")
        def _status(self, _):
            pass

        def toggle(self):
            if self.recording:
                self._stop()
            else:
                self._start()

        def _start(self):
            self.title = "● Recording"
            self.menu[0].title = "Status: recording"
            self.capture = AudioCapture()
            self.capture.start()
            self.recording = True

        def _stop(self):
            self.title = "VoiceEngine"
            self.menu[0].title = "Status: transcribing…"
            self.recording = False
            if self.capture:
                self.capture.stop()
                audio = self.capture.take_buffer()
                self.capture = None
                import threading
                threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

        def _transcribe(self, audio):
            if audio is None or len(audio) == 0:
                self.menu[0].title = "Status: idle"
                if hasattr(self, 'title'):
                    self.title = "VoiceEngine"
                return
            text, timing = self.stt.transcribe(audio, return_timing=True)
            print(json.dumps({"text": text, "total_ms": round(timing["total"], 1)}))
            import subprocess
            # Paste via AppleScript
            subprocess.run([
                "osascript", "-e",
                f'''set the clipboard to "{text.replace(chr(34), "")}"
                tell application "System Events" to keystroke "v" using command down'''
            ])
            self.menu[0].title = "Status: idle"

except ImportError:
    rumps = None
    VoiceApp = None

# ── CLI Entry Point ───────────────────────────────────────────────────────────

def load_audio(path):
    import numpy as np
    with wave.open(path) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == SR, f"Expected 16kHz, got {w.getframerate()}"
        n = w.getnframes()
        raw = w.readframes(n)
        return np.array(struct.unpack("<" + "h" * n, raw), dtype=np.float32) / 32768.0

def main_cli():
    import argparse
    parser = argparse.ArgumentParser(description="VoiceEngine STT")
    parser.add_argument("--file", type=str, help="Transcribe a WAV file")
    parser.add_argument("--record", action="store_true", help="Record from mic and transcribe")
    parser.add_argument("--app", action="store_true", help="Menubar app with Caps Lock toggle")
    args = parser.parse_args()

    stt = MoonshineSTT()
    stt.load()

    if args.app:
        if rumps is None:
            print("Install rumps: uv pip install rumps")
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "rumps", "-q"])
            from rumps import App as rumps_App
            rumps.App = rumps_App
        app = VoiceApp(stt)
        monitor = CapsLockMonitor(on_press=app.toggle, on_release=app.toggle)
        monitor.start()
        print("Menubar app running. Press Caps Lock to dictate.")
        app.run()
    elif args.file:
        audio = load_audio(args.file)
        text, timing = stt.transcribe(audio, return_timing=True)
        print(json.dumps({
            "text": text,
            "encoder_ms": round(timing["encoder"], 1),
            "kv_ms": round(timing["kv"], 1),
            "decoder_ms": round(timing["decoder"], 1),
            "total_ms": round(timing["total"], 1),
        }, indent=2))
    elif args.record:
        print("Hold Caps Lock to record. Release to transcribe. Ctrl+C to quit.")
        import threading
        action = [None]

        def on_press():
            c = AudioCapture()
            c.start()
            action[0] = c

        def on_release():
            if action[0]:
                action[0].stop()
                audio = action[0].take_buffer()
                action[0] = None
                if audio is not None and len(audio) > 0:
                    t = stt.transcribe(audio)
                    print(json.dumps({"text": t}))
                    # Paste via AppleScript
                    import subprocess
                    escaped = t.replace('"', '\\"')
                    subprocess.run(["osascript", "-e",
                        f'set the clipboard to "{escaped}"'
                    ], input=b'tell application "System Events" to keystroke "v" using command down',
                        capture_output=True, text=True)

        monitor = CapsLockMonitor(on_press=on_press, on_release=on_release)
        monitor.start()
        import Quartz
        print("  (Caps Lock down = record, up = transcribe + paste)")
        try:
            CG.CFRunLoopRun()
        except KeyboardInterrupt:
            if rec[0] and action[0]:
                action[0].stop()
        audio = cap.take_buffer()
        if audio is None or len(audio) == 0:
            print("No audio captured")
            return
        text, timing = stt.transcribe(audio, return_timing=True)
        print(json.dumps({
            "text": text,
            "encoder_ms": round(timing["encoder"], 1),
            "kv_ms": round(timing["kv"], 1),
            "decoder_ms": round(timing["decoder"], 1),
            "total_ms": round(timing["total"], 1),
        }, indent=2))

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv.append("--record")
    main_cli()
