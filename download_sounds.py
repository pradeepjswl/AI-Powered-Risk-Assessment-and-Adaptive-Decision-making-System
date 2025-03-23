import os
import requests
import urllib.request
from pathlib import Path

def download_sound(url, filename):
    """
    Download a sound file from URL and save it to the sounds directory
    """
    try:
        # Create sounds directory if it doesn't exist
        os.makedirs('sounds', exist_ok=True)
        
        # Full path for the sound file
        filepath = os.path.join('sounds', filename)
        
        # Download the file
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def main():
    # Sound file URLs (using free sound effects from a reliable source)
    sound_urls = {
        'vehicle_alert.mp3': 'https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3',  # Car horn
        'person_alert.mp3': 'https://assets.mixkit.co/active_storage/sfx/2868/2868-preview.mp3',   # Warning beep
        'danger_alert.mp3': 'https://assets.mixkit.co/active_storage/sfx/2867/2867-preview.mp3',   # Emergency alert
        'behavior_alert.mp3': 'https://assets.mixkit.co/active_storage/sfx/2866/2866-preview.mp3'  # Alert tone
    }
    
    print("Starting sound file downloads...")
    
    # Download each sound file
    for filename, url in sound_urls.items():
        download_sound(url, filename)
    
    print("\nSound file download process completed!")
    print("\nPlease check the 'sounds' directory for the downloaded files:")
    for filename in sound_urls.keys():
        filepath = os.path.join('sounds', filename)
        if os.path.exists(filepath):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} (Failed to download)")

if __name__ == "__main__":
    main() 