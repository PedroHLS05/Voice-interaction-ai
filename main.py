import threading
import queue
import unicodedata
import time

import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import speech_recognition as sr
import pyttsx3
import numpy as np
import pyaudio
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Inicialização do sintetizador de voz
engine = pyttsx3.init()


def _speak_blocking(text):
    engine.say(text)
    engine.runAndWait()


def speak(text):
    t = threading.Thread(target=_speak_blocking, args=(text,), daemon=True)
    t.start()


def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")


class AudioCapture(threading.Thread):
    def __init__(self, buffer_queue: queue.Queue, stop_event: threading.Event, rate=44100, frames_per_buffer=1024):
        super().__init__(daemon=True)
        self.buffer_queue = buffer_queue
        self.stop_event = stop_event
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.p = None
        self.stream = None

    def run(self):
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True,
                                      frames_per_buffer=self.frames_per_buffer)
            while not self.stop_event.is_set():
                try:
                    data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
                except Exception:
                    continue
                samples = np.frombuffer(data, dtype=np.int16)
                self.buffer_queue.put(samples)
        finally:
            if self.stream is not None:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
            if self.p is not None:
                try:
                    self.p.terminate()
                except Exception:
                    pass


class RecognizerThread(threading.Thread):
    def __init__(self, text_queue: queue.Queue, stop_event: threading.Event, language="pt-BR"):
        super().__init__(daemon=True)
        self.text_queue = text_queue
        self.stop_event = stop_event
        self.language = language
        self.recognizer = sr.Recognizer()

    def run(self):
        mic = sr.Microphone()
        with mic as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            except Exception:
                pass

        while not self.stop_event.is_set():
            with mic as source:
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                except sr.WaitTimeoutError:
                    continue
                except Exception:
                    continue
                try:
                    text = self.recognizer.recognize_google(audio, language=self.language)
                    self.text_queue.put(text)
                except sr.UnknownValueError:
                    self.text_queue.put(None)
                except sr.RequestError as e:
                    self.text_queue.put(f"[ERRO REQ] {e}")


class VoiceUI:
    def __init__(self, root):
        self.root = root
        root.title("Arena — Voice Interaction")
        self.running = False

        # Plot setup
        self.fig = Figure(figsize=(8, 3))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Ondas Sonoras em Tempo Real")
        self.ax.set_xlabel("Amostras")
        self.ax.set_ylabel("Amplitude")
        self.line, = self.ax.plot([], [])

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Texto transcrito
        self.text_display = ScrolledText(root, height=8)
        self.text_display.pack(fill=tk.BOTH, expand=False)

        # Botões
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X)
        self.start_btn = tk.Button(btn_frame, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.quit_btn = tk.Button(btn_frame, text="Quit", command=self.quit)
        self.quit_btn.pack(side=tk.RIGHT, padx=5, pady=5)

        # Buffers e threads
        self.buffer_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.audio_thread = None
        self.rec_thread = None

        # Activation state
        self.activated = False
        self.last_activation_time = 0
        self.activation_timeout = 12.0

        # Ring buffer for display
        self.display_buffer = np.zeros(44100 // 8, dtype=np.int16)  # ~0.125s buffer
        self.update_plot()

    def start(self):
        if self.running:
            return
        self.running = True
        self.stop_event.clear()
        self.audio_thread = AudioCapture(self.buffer_queue, self.stop_event)
        self.rec_thread = RecognizerThread(self.text_queue, self.stop_event)
        self.audio_thread.start()
        self.rec_thread.start()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        speak("Arena pronta. Diga 'Olá Arena' para ativar.")

    def stop(self):
        if not self.running:
            return
        self.stop_event.set()
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def quit(self):
        self.stop()
        self.root.after(200, self.root.destroy)

    def update_plot(self):
        # Push queued audio into display buffer
        while not self.buffer_queue.empty():
            samples = self.buffer_queue.get()
            n = len(samples)
            if n >= len(self.display_buffer):
                self.display_buffer = samples[-len(self.display_buffer):]
            else:
                self.display_buffer = np.roll(self.display_buffer, -n)
                self.display_buffer[-n:] = samples

        # Atualiza linha do plot
        self.line.set_data(np.arange(len(self.display_buffer)), self.display_buffer)
        self.ax.set_xlim(0, len(self.display_buffer))
        minv = int(np.min(self.display_buffer) - 100) if self.display_buffer.size else -32768
        maxv = int(np.max(self.display_buffer) + 100) if self.display_buffer.size else 32767
        if minv == maxv:
            minv -= 1
            maxv += 1
        self.ax.set_ylim(minv, maxv)
        self.canvas.draw_idle()

        # Processa texto reconhecido
        while not self.text_queue.empty():
            txt = self.text_queue.get()
            if txt is None:
                # Não entendeu
                self.text_display.insert(tk.END, "[Não entendi]\n")
                self.text_display.see(tk.END)
                continue
            if isinstance(txt, str) and txt.startswith("[ERRO REQ]"):
                self.text_display.insert(tk.END, txt + "\n")
                self.text_display.see(tk.END)
                continue

            display_line = f"Você disse: {txt}\n"
            self.text_display.insert(tk.END, display_line)
            self.text_display.see(tk.END)

            normalized = _strip_accents(txt.lower())
            if not self.activated:
                if "ola arena" in normalized or "ola, arena" in normalized or "olaarena" in normalized or "ola arena" in normalized:
                    self.activated = True
                    self.last_activation_time = time.time()
                    self.text_display.insert(tk.END, "[Arena ativada]\n")
                    self.text_display.see(tk.END)
                    speak("Olá, eu sou a Arena. Em que posso ajudar?")
            else:
                # Já ativada — interpretar comandos
                self.last_activation_time = time.time()
                if "sair" in normalized or "tchau" in normalized or "ate logo" in normalized:
                    speak("Saindo... Até logo!")
                    self.quit()
                elif "como voce esta" in normalized or "como voce está" in txt.lower():
                    speak("Eu estou ótima! E você?")
                else:
                    speak(f"Você disse: {txt}")

        # Auto-desativar se inativa por muito tempo
        if self.activated and (time.time() - self.last_activation_time) > self.activation_timeout:
            self.activated = False
            self.text_display.insert(tk.END, "[Arena desativada por inatividade]\n")
            self.text_display.see(tk.END)

        # schedule next update
        self.root.after(50, self.update_plot)


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceUI(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()