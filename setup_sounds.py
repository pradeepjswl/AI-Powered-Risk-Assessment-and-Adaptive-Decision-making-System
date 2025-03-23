import os
import sys
from download_sounds import download_sound
from generate_alert_sounds import generate_alert_sound

def setup_sounds():
    """
    Set up sound files by trying to download them first, then generating if needed
    """
    # Sound file URLs
    sound_urls = {
        'vehicle_alert.mp3': 'https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3',
        'person_alert.mp3': 'https://assets.mixkit.co/active_storage/sfx/2868/2868-preview.mp3',
        'danger_alert.mp3': 'https://assets.mixkit.co/active_storage/sfx/2867/2867-preview.mp3',
        'behavior_alert.mp3': 'https://assets.mixkit.co/active_storage/sfx/2866/2866-preview.mp3'
    }
    
    # Create sounds directory
    os.makedirs('sounds', exist_ok=True)
    
    print("Setting up sound files...")
    
    # Try to download each sound file
    for filename, url in sound_urls.items():
        filepath = os.path.join('sounds', filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"✓ {filename} already exists")
            continue
            
        print(f"Attempting to download {filename}...")
        if download_sound(url, filename):
            print(f"✓ Successfully downloaded {filename}")
        else:
            print(f"✗ Failed to download {filename}, generating backup sound...")
            # Generate backup sound with appropriate frequency
            if filename == 'vehicle_alert.mp3':
                generate_alert_sound(filepath, 440)  # A4 note
            elif filename == 'person_alert.mp3':
                generate_alert_sound(filepath, 880)  # A5 note
            elif filename == 'danger_alert.mp3':
                generate_alert_sound(filepath, 660)  # E5 note
            elif filename == 'behavior_alert.mp3':
                generate_alert_sound(filepath, 550)  # C#5 note
            print(f"✓ Generated backup sound for {filename}")
    
    # Verify all files exist
    print("\nVerifying sound files...")
    all_files_exist = True
    for filename in sound_urls.keys():
        filepath = os.path.join('sounds', filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} is ready")
        else:
            print(f"✗ {filename} is missing")
            all_files_exist = False
    
    if all_files_exist:
        print("\nAll sound files are ready!")
    else:
        print("\nSome sound files are missing. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    setup_sounds() 