from playsound import playsound
import threading


def _play_sound(path):
    threading.Thread(
        target=lambda: playsound(path),
        daemon=True
    ).start()


def drowsy_alert():
    """
    Mild warning sound
    """
    _play_sound("../assets/beep.wav")


def critical_alert():
    """
    Loud alarm for microsleep
    """
    _play_sound("../assets/alarm.wav")
