import sounddevice as sd
from scipy.io.wavfile import write, read
import numpy as np
import threading
import queue


def record_audio(file_name="temp.wav", sample_rate=44100):
    """
    Graba audio desde el micrófono y lo guarda en un archivo WAV.
    La grabación se detiene al presionar Enter.

    Args:
        file_name (str): Nombre del archivo WAV de salida.
        sample_rate (int): Frecuencia de muestreo en Hz.
    """
    # Cola para almacenar los fragmentos de audio
    q = queue.Queue()

    # Variable para controlar la grabación
    recording = True

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    def stop_recording():
        input("Presiona Enter para detener la grabación...\n")
        nonlocal recording
        recording = False

    print("Grabando... Presiona Enter para detener.")

    # Inicia el hilo para detectar cuando se presiona Enter
    threading.Thread(target=stop_recording, daemon=True).start()

    # Prepara el stream de audio
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        chunks = []
        while recording:
            chunks.append(q.get())

    # Combina todos los chunks de audio
    audio_data = np.concatenate(chunks)

    print("Grabación finalizada. Guardando el archivo...")
    write(file_name, sample_rate, audio_data)
    print(f"Archivo guardado como {file_name}")


def play_audio(file_name="temp.wav"):
    """
    Reproduce un archivo de audio.

    Args:
        file_name (str): Nombre del archivo WAV a reproducir.
    """
    try:
        # Lee el archivo WAV
        sample_rate, audio_data = read(file_name)

        # Reproduce el audio
        print(f"Reproduciendo {file_name}...")
        sd.play(audio_data, sample_rate)
        sd.wait()  # Espera hasta que termine la reproducción
        print("Reproducción finalizada.")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {file_name}")
    except Exception as e:
        print(f"Error al reproducir el audio: {e}")


if __name__ == "__main__":
    try:
        file_name = input(
            "Ingresa el nombre del archivo de salida (por defecto: output.wav): ") or "output.wav"
        record_audio(file_name)
    except Exception as e:
        print("Ocurrió un error:", e)
