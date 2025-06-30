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
    "emo_mapping_wplus_2.pt": "17C1-ACpPbFnaRNVYpDPrzNTYbFJPL_7h",
    "resnet50_ft_weight.pkl": "1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU",
    "model_ir_se50.pth": "1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn",
}
# Direct URLs
other_files = {
    "emonet_8.pth": "https://github.com/face-analysis/emonet/raw/refs/heads/master/pretrained/emonet_8.pth",
    "ffhq.pkl": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
    # "shape_predictor_68_face_landmarks.dat.bz2": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
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

# Download other files (direct URLs)
for filename, url in other_files.items():
    file_path = os.path.join(pretrained_dir, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"{filename} downloaded successfully.")
    else:
        print(f"{filename} already exists, skipping download.")

print("All files downloaded successfully.")
