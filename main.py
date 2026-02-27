import speech_recognition as sr
import pyttsx3
import numpy as np
import matplotlib.pyplot as plt
import pyaudio

# Inicialização do sintetizador de voz
engine = pyttsx3.init()

# Função para exibir a onda sonora
def plot_waveform(audio_data):
    plt.clf()  # Limpa a figura antes de redesenhar
    plt.plot(audio_data)
    plt.title("Ondas Sonoras em Tempo Real")
    plt.xlabel("Tempo")
    plt.ylabel("Amplitude")
    plt.draw()
    plt.pause(0.01)

# Função para ouvir e reconhecer a voz
def listen_and_respond():
    recognizer = sr.Recognizer()

    # Configurar microfone
    with sr.Microphone() as source:
        print("Aguardando fala...")
        recognizer.adjust_for_ambient_noise(source)  # Ajusta para o ruído ambiente
        audio = recognizer.listen(source)

        # Convertendo o áudio em texto
        try:
            print("Reconhecendo...")
            speech_text = recognizer.recognize_google(audio)
            print("Você disse: " + speech_text)
            return speech_text
        except sr.UnknownValueError:
            print("Não consegui entender o que você disse.")
            return None
        except sr.RequestError as e:
            print(f"Erro na requisição ao serviço de reconhecimento de voz: {e}")
            return None

# Função para sintetizar a fala da IA
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Função para capturar áudio em tempo real e exibir as ondas sonoras
def capture_audio_and_display_waveform():
    p = pyaudio.PyAudio()

    # Abrir fluxo de áudio
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    print("Capturando áudio...")

    plt.ion()  # Modo interativo para plotagem
    while True:
        data = np.frombuffer(stream.read(1024), dtype=np.int16)  # Captura o áudio em tempo real
        plot_waveform(data)  # Exibe a onda sonora

# Função principal
def main():
    plt.figure(figsize=(10, 4))  # Tamanho da figura para a onda sonora
    speak("Olá! Como posso ajudar?")  # Fala inicial da IA

    while True:
        user_input = listen_and_respond()  # Captura a fala do usuário

        if user_input:
            if "sair" in user_input.lower():
                speak("Saindo... Até logo!")
                break
            elif "como você está" in user_input.lower():
                speak("Eu estou ótimo! E você?")
            else:
                speak(f"Você disse: {user_input}")

# Rodar as funções em paralelo (captura de áudio e interação com o usuário)
import threading

if __name__ == "__main__":
    # Thread para captura de áudio em tempo real e exibição das ondas
    audio_thread = threading.Thread(target=capture_audio_and_display_waveform)
    audio_thread.daemon = True
    audio_thread.start()

    # Rodar o fluxo principal de interação com o usuário
    main()