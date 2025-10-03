#!/usr/bin/env bash
cd "$(dirname "$0")"
set -e  # Exit on any error

# Check if arguments are provided
# setup_giqa.sh <gpu_id> <path_to_training_images>
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <gpu_id> <path_to_training_images>"
    echo "Example: $0 0 /path/to/FFHQ/images1024x1024/"
    exit 1
else
    GPU_ID=$1
    TRAINING_IMAGES_PATH=$2
    if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
        echo "Error: GPU ID must be a number"
        exit 1
    fi
    if [ ! -d "$TRAINING_IMAGES_PATH" ]; then
        echo "Error: Training images directory does not exist: $TRAINING_IMAGES_PATH"
        exit 1
    fi
fi

GIQA_DIR="vendor/GIQA"
WEIGHTS_DIR="$GIQA_DIR/weights"

# Clone GIQA repository
echo "Cloning GIQA repository..."
rm -rf "$GIQA_DIR"
git clone https://github.com/cientgu/GIQA "$GIQA_DIR"

# Patch GIQA code
echo "Applying small patches to GIQA code..."
# Make image file fetching recursive
sed -i \
    -e "s|path.glob('\*.jpg')|path.glob('\*\*/\*.jpg')|g" \
    -e "s|path.glob('\*.png')|path.glob('\*\*/\*.png')|g" \
    "$GIQA_DIR/write_act.py"

# Extract features from the images of the FFHQ training dataset
echo "Extracting features from training images..."
mkdir -p "$WEIGHTS_DIR"
python "$GIQA_DIR/write_act.py" \
  "$TRAINING_IMAGES_PATH" \
  --act_path "$WEIGHTS_DIR/acts.pth" \
  --gpu "$GPU_ID"

# Fit a Gaussian Mixture Model to these features
echo "Fitting Gaussian Mixture Model..."
python "$GIQA_DIR/get_gmm.py" \
  --act_path "$WEIGHTS_DIR/acts.pth" \
  --kernel_number 10 \
  --gmm_path "$WEIGHTS_DIR/gmm_model.pth"

# Remove extracted features to save space
echo "Cleaning up extracted features..."
rm "$WEIGHTS_DIR/acts.pth"

echo "GIQA setup completed! Weights are saved in '$WEIGHTS_DIR'."
