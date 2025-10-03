import os
import gdown
import requests


# Create pretrained directory if it does not exist
pretrained_dir = 'pretrained'
if not os.path.exists(pretrained_dir):
    os.makedirs(pretrained_dir)

# List of files to download
# Google Drive IDs
google_drive_files = {
    "magface_epoch_00025.pth": "1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H",
}

# Download google drive files
for filename, id in google_drive_files.items():
    url = f"https://drive.google.com/uc?id={id}"
    file_path = os.path.join(pretrained_dir, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(url, file_path, quiet=False)
    else:
        print(f"{filename} already exists, skipping download.")

print("All files downloaded successfully.")
