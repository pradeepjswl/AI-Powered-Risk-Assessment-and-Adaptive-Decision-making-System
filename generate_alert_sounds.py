from pydub import AudioSegment
from pydub.generators import Sine
import os

def generate_alert_sound(filename, frequency, duration_ms=1000, volume=-20):
    """
    Generate an alert sound file using pydub
    """
    # Generate a sine wave
    sound = Sine(frequency).to_audio_segment(duration=duration_ms)
    
    # Add some modulation for more alert-like sound
    modulated = sound.overlay(Sine(frequency * 1.5).to_audio_segment(duration=duration_ms))
    
    # Normalize and set volume
    normalized = modulated.normalize()
    normalized = normalized - abs(volume)  # Reduce volume
    
    # Export as MP3
    normalized.export(filename, format="mp3")

def main():
    # Create sounds directory if it doesn't exist
    os.makedirs('sounds', exist_ok=True)
    
    # Generate different alert sounds with distinct frequencies
    generate_alert_sound('sounds/vehicle_alert.mp3', 440)  # A4 note
    generate_alert_sound('sounds/person_alert.mp3', 880)  # A5 note
    generate_alert_sound('sounds/danger_alert.mp3', 660)  # E5 note
    generate_alert_sound('sounds/behavior_alert.mp3', 550)  # C#5 note
    
    print("Alert sound files generated successfully!")

if __name__ == "__main__":
    main() 