import os
# Set working dir to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
import shutil
import argparse
from dataclasses import dataclass
import random
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'vendor/emostyle'))
from vendor.emostyle.generate_dataset_pkl import generate_data as stylegan2
from vendor.emostyle.models.emo_mapping import EmoMappingWplus
from vendor.emostyle.models.stylegan2_interface import StyleGAN2
from vendor.emostyle.models.emonet import EmoNet

from src.filter import calculate_threshold
from src.quality.giqa import GIQA
from src.quality.magface import MagFace


@dataclass
class Emotion:
    valence: float
    arousal: float

# Define the emotions
EMOTIONS = {
    "Happy":        Emotion(0.991, 0.136),
    "Delighted":    Emotion(0.907, 0.421),
    "Excited":      Emotion(0.661, 0.75),
    "Astonished":   Emotion(0.345, 0.938),
    "Aroused":      Emotion(0.279, 0.96),
    "Tense":        Emotion(-0.049, 0.999),
    "Alarmed":      Emotion(-0.113, 0.994),
    "Angry":        Emotion(-0.156, 0.988),
    "Afraid":       Emotion(-0.438, 0.899),
    "Annoyed":      Emotion(-0.545, 0.839),
    "Distressed":   Emotion(-0.743, 0.669),
    "Frustrated":   Emotion(-0.777, 0.629),
    "Miserable":    Emotion(-0.988, -0.151),
    "Sad":          Emotion(-0.887, -0.462),
    "Gloomy":       Emotion(-0.875, -0.485),
    "Depressed":    Emotion(-0.857, -0.515),
    "Bored":        Emotion(-0.469, -0.883),
    "Droopy":       Emotion(-0.23, -0.973),
    "Tired":        Emotion(-0.04, -0.999),
    "Sleepy":       Emotion(0.033, -0.999),
    "Calm":         Emotion(0.722, -0.692),
    "Relaxed":      Emotion(0.743, -0.669),
    "Satisfied":    Emotion(0.755, -0.656),
    "AtEase":       Emotion(0.777, -0.629),
    "Content":      Emotion(0.799, -0.602),
    "Serene":       Emotion(0.854, -0.521),
    "Glad":         Emotion(0.982, -0.191),
    "Pleased":      Emotion(0.993, -0.118),
}

# Default GPU ID
GPU_ID = 0  # GPU selection is not supported yet 
# Internal constants
EMO_EMBED = 64
STG_EMBED = 512
INPUT_SIZE = 1024
ITER_NUM = 500
FILTER_INPUT_SIZE = 112
MAX_ITERATION_COUNT = 10  # Maximum iterations for quality control loop
# Temporary directories
TMP_DIR = "tmp"
TMP_INPUT_DIR = "tmp/input"
TMP_FILTER_INPUT_DIR = "tmp/input/small"
# Model weights paths
STYLEGAN2_WEIGHTS_PATH = "pretrained/ffhq.pkl"
EMOSTYLE_WEIGHTS_PATH = "pretrained/emo_mapping_wplus_2.pt"
EMONET_WEIGHTS_PATH = "pretrained/emonet_8.pth"
GIQA_WEIGHTS_PATH = "vendor/GIQA/weights/gmm_model.pth"
MAGFACE_WEIGHTS_PATH = "pretrained/magface_epoch_00025.pth"
# Script paths
MAGFACE_SCRIPT_PATH = "vendor/MagFace/inference/gen_feat.py"


def generate_random_image_set(
    output_dir: str,
    count: int = 1,
    force_cpu: bool = False
) -> None:
    """
    Generate a set of random images and save them to the specified output directory.
    Note: StyleGAN2 may generate more images than requested due to batching. Extra images are removed.
    """
    # Define generation batch size
    batch_size = 8
    if batch_size > count:
        batch_size = count  # Ensure batch size is not more than the number of samples
    
    # Generate random images using StyleGAN2
    stylegan2(
        checkpoint_path=STYLEGAN2_WEIGHTS_PATH,
        outdir=output_dir,
        n_samples=count,
        batch_size=batch_size,
        truncation_psi=0.7,
        on_cpu=force_cpu
    )
    
    # StyleGAN2 may generate extra images due to batching, remove excess
    generated_images = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    if len(generated_images) > count:
        # Remove excess images (keep the first 'count' images)
        for img_file in generated_images[count:]:
            img_path = os.path.join(output_dir, img_file)
            npy_path = os.path.splitext(img_path)[0] + '.npy'
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(npy_path):
                os.remove(npy_path)


class EmoStyle:
    """
    A class to handle the generation of images with specific emotions using the EmoStyle framework.
    """

    def __init__(self, force_cpu: bool = False):
        self.device = 'cuda' if not force_cpu and torch.cuda.is_available() else 'cpu'
        self.stylegan = None
        self.emonet = None
        self.emo_mapping = None

        # Load the necessary models and mappings
        self._load_models()


    def _load_models(self):
        """
        Load the necessary models and mappings for the EmoStyle framework.
        """
        # Load the StyleGAN2 model
        self.stylegan = StyleGAN2(
            checkpoint_path=STYLEGAN2_WEIGHTS_PATH,
            stylegan_size=INPUT_SIZE,
            is_dnn=True,
            is_pkl=True
        )
        self.stylegan.eval()
        self.stylegan.requires_grad_(False)
        
        # Load the EmoMapping model
        ckpt_emo_mapping = torch.load(EMOSTYLE_WEIGHTS_PATH, map_location=self.device)
        self.emo_mapping = EmoMappingWplus(INPUT_SIZE, EMO_EMBED, STG_EMBED)
        self.emo_mapping.load_state_dict(ckpt_emo_mapping['emo_mapping_state_dict'])
        self.emo_mapping.eval()

        # Move models to the appropriate device (CPU or GPU)
        self.emo_mapping = self.emo_mapping.to(self.device)
        self.stylegan = self.stylegan.to(self.device)


    def edit_image(self, image_path: str, emotions: list[Emotion], output_dir: str) -> None:
        """
        Edit the input image with for each of the specified emotions and save the output images to the output path.
        """
        # Find image latent vector from the input image path, then load it.
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        latent_path = os.path.splitext(image_path)[0] + '.npy'
        image_latent = np.load(latent_path, allow_pickle=False)
        image_latent = np.expand_dims(image_latent[:, :], 0)
        latent = torch.from_numpy(image_latent).float().to(self.device)
        
        # For every combination of angle and strength, generate and save an output image.
        for emotion in tqdm(emotions, desc=f"Editing {image_name}"):
            emotion_vector = torch.FloatTensor([[emotion.valence, emotion.arousal]]).to(self.device)
            fake_latents = latent + self.emo_mapping(latent, emotion_vector)
            generated_image_tensor = self.stylegan.generate(fake_latents)
            generated_image_tensor = (generated_image_tensor + 1.) / 2.
            generated_image = generated_image_tensor.detach().cpu().squeeze().numpy()
            generated_image = np.clip(generated_image * 255, 0, 255).astype(np.uint8)
            generated_image = generated_image.transpose(1, 2, 0)
            
            # out_filename = os.path.join(output_path, f"{image_name}_{angle}_{strength}.png")
            out_filename = os.path.join(output_dir, f"{image_name}_va_{emotion.valence:.3f}_{emotion.arousal:.3f}.png")
            plt.imsave(out_filename, generated_image)


def _resume_next_and_last_image_ids(tmp_dir: str) -> tuple[int, int]:
    """
    Returns the first image ID to resume from or -1 if no images are found in the tmp directory.
    """
    if os.path.exists(tmp_dir):
        existing_images = [f for f in os.listdir(tmp_dir) if f.endswith('.png')]
        if existing_images:
            existing_images.sort()
            first_image = existing_images[0]
            last_image = existing_images[-1]
            try:
                first_image_id = int(os.path.splitext(first_image)[0])
                last_image_id = int(os.path.splitext(last_image)[0])
                return first_image_id, last_image_id
            except ValueError:
                raise ValueError(f"Could not find from where to resume. Invalid image file name '{first_image}'. Expected format: {{image_id:06}}.png")
        else:
            print("Could not find from where to resume. No existing images found in the tmp directory.")
    else:
        print("Could not find from where to resume. No tmp directory found.")
    
    return -1, -1

def _resume_next_emotion_index(output_dir: str, first_image_id: int) -> int:
    """
    Returns the first emotion index to resume from.
    """
    # Find generated image with the lastly generated image ID
    if os.path.exists(output_dir):
        existing_images = [f for f in os.listdir(output_dir) if f.startswith(f"{first_image_id:06}_va_")]
        # The next index to generate is the number of images already generated
        if existing_images:
            return len(existing_images)
    else:
        print("Could not find from where to resume. No output directory found.")
    
    return 0


def _apply_quality_control(
    input_dir: str,
    quality_method: str,
    quality_requirement: str,
    force_cpu: bool,
    tmp_dir: str
) -> int:
    """
    Apply quality control to base face images. Returns the number of images that passed quality control.
    
    For percentage-based quality requirements, generates extra images upfront and keeps only the best ones.
    For absolute quality requirements, uses an iterative approach with MAX_ITERATION_COUNT limit.
    """
    # Setup quality assessment method
    if quality_method == "giqa":
        quality_filter = GIQA(GIQA_WEIGHTS_PATH, tmp_dir, -1 if force_cpu else GPU_ID)
    elif quality_method == "magface":
        quality_filter = MagFace(MAGFACE_SCRIPT_PATH, MAGFACE_WEIGHTS_PATH, tmp_dir, -1 if force_cpu else GPU_ID)
    else:
        raise ValueError(f"Invalid quality method '{quality_method}'. Available methods: 'giqa', 'magface'.")
    
    # Determine if percentage or absolute quality requirement
    is_percentage = quality_requirement.endswith('%')
    
    if is_percentage:
        # Percentage-based: generate extra images upfront and keep the best ones
        percentage = float(quality_requirement[:-1])
        if not (0 <= percentage <= 100):
            raise ValueError(f"Percentage must be between 0 and 100, got {percentage}")
        
        print(f"Applying {quality_method.upper()} quality control to keep top {100-percentage}% of base face images...")
        
        # Calculate quality scores for all base images
        quality_scores = quality_filter.calculate_quality_scores(input_dir)
        if not quality_scores:
            print("No images found for quality assessment.")
            return 0
        
        # Calculate threshold as the percentile to remove bottom X% of images
        threshold_value = calculate_threshold(quality_scores, percentage)
        
        # Remove images below threshold
        removed_count = 0
        for image_path, score in quality_scores.items():
            if score < threshold_value:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    # Also remove corresponding .npy file
                    npy_path = os.path.splitext(image_path)[0] + '.npy'
                    if os.path.exists(npy_path):
                        os.remove(npy_path)
                    removed_count += 1
        
        kept_count = len(quality_scores) - removed_count
        print(f"Quality control: assessed {len(quality_scores)} images, kept {kept_count} (top {100-percentage}%), removed {removed_count} (bottom {percentage}%)")
        print(f"Quality threshold: {threshold_value:.2f}")
        return kept_count
    
    else:
        # Absolute quality requirement: iterative approach
        threshold_value = float(quality_requirement)
        print(f"Applying {quality_method.upper()} quality control with minimum score {threshold_value}...")
        
        iteration = 0
        total_generated = 0
        total_removed = 0
        
        while iteration < MAX_ITERATION_COUNT:
            iteration += 1
            
            # Calculate quality scores for current images
            quality_scores = quality_filter.calculate_quality_scores(input_dir)
            if not quality_scores:
                print(f"Iteration {iteration}: No images found.")
                break
            
            # Find images below threshold
            below_threshold = [(img, score) for img, score in quality_scores.items() if score < threshold_value]
            if not below_threshold:
                print(f"Iteration {iteration}: All {len(quality_scores)} images meet quality requirement (score >= {threshold_value})")
                break
            
            print(f"Iteration {iteration}: {len(below_threshold)} out of {len(quality_scores)} images below threshold")
            
            # Remove poor quality images
            for image_path, score in below_threshold:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    # Also remove corresponding .npy file
                    npy_path = os.path.splitext(image_path)[0] + '.npy'
                    if os.path.exists(npy_path):
                        os.remove(npy_path)
                    total_removed += 1
            
            # Generate replacement images
            replacement_count = len(below_threshold)
            print(f"Generating {replacement_count} replacement images...")
            generate_random_image_set(input_dir, count=replacement_count, force_cpu=force_cpu)
            total_generated += replacement_count
            
            if iteration >= MAX_ITERATION_COUNT:
                print(f"Warning: Reached maximum iteration count ({MAX_ITERATION_COUNT})")
                # Calculate final quality scores
                final_scores = quality_filter.calculate_quality_scores(input_dir)
                passing = sum(1 for score in final_scores.values() if score >= threshold_value)
                print(f"Final result: {passing} out of {len(final_scores)} images meet quality requirement")
                return len(final_scores)
        
        final_count = len([f for f in os.listdir(input_dir) if f.endswith('.png')])
        print(f"Quality control complete: {total_generated} total replacements generated, {total_removed} removed")
        return final_count




# --- Main --------------------------------------------------------------------
if __name__ == "__main__":   
    # generate.py [-n num_faces] [-o output_dir] [-c] [-m method] [-q quality] random [fe_count] [magnitude_coeff]
    # generate.py [-n num_faces] [-o output_dir] [-c] [-m method] [-q quality] emotions <emotion1 [emotion2 ...]> [-s strengths...] 
    # generate.py [-n num_faces] [-o output_dir] [-c] [-m method] [-q quality] va <valence0 arousal0 [valence1 arousal1 ...]>
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate images with specific facial expressions.")
    parser.add_argument("-n", "--num_faces", type=int, default=1,
                        help="Number of unique face identities to generate (default: 1).")
    parser.add_argument("-o", "--output_dir", type=str, default="output",
                        help="Directory to save the generated images (default: 'output').")
    parser.add_argument("-c", "--force_cpu", action="store_true",
                        help="Force the program to run on CPU, even if a GPU is available.")
    parser.add_argument("-r", "--resume", action="store_true",
                        help="Resume from the last generated image. "
                        "This will skip already generated images based on the content of the output and tmp directories.")
    parser.add_argument("-m", "--method", default="giqa", type=str, choices=["giqa", "magface"],
                        help="Quality assessment method to use for quality control (default: 'giqa').")
    parser.add_argument("-q", "--quality", type=str, default=None,
                        help="Quality requirement for base face images. Can be an absolute score (e.g., '50.0') or a percentage (e.g., '20%%') "
                        "to keep only the top (100-X)%% of images. If not specified, no quality control is applied.")

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    # Random mode
    random_parser = subparsers.add_parser("random", help="Generate random valence-arousal pairs.")
    random_parser.add_argument("fe_count", type=int, default=1, nargs="?",
                               help="Number of random facial expressions to generate per face (default: 1).")
    random_parser.add_argument("magnitude_coeff", type=float, default=0.0, nargs="?",
                               help="Magnitude coefficient to adjust the strength of random emotions (0.0 to 1.0).")

    # Emotions mode
    emotions_parser = subparsers.add_parser("emotions", help="Apply predefined emotions.")
    emotions_parser.add_argument("emotions", type=str, nargs="+",
                                help="List of predefined emotions to apply. Available emotions: " + ", ".join(EMOTIONS.keys()) + ".")
    emotions_parser.add_argument("-s", "--strengths", type=float, nargs="+", default=[1.0],
                                help="List of strengths (from 0.0 to 1.0) for each emotion (default: 1.0 for all emotions).")
    emotions_parser.add_argument("-u", "--unique", action="store_true",
                                help="Use a different face for each generated image, "
                                "which means that `--num_faces` must be a multiple of `len(strengths) * len(emotions)`. "
                                "If not set, the same face will be used once for each combination of emotion and strength.")

    # Valence-Arousal mode
    va_parser = subparsers.add_parser("va", help="Apply custom valence-arousal pairs.")
    va_parser.add_argument("valences_arousals", type=float, nargs="+",
                        help="List of valence-arousal pairs (valence0 arousal0 valence1 arousal1 ...). Each pair specifies a custom emotion.")
    va_parser.add_argument("-u", "--unique", action="store_true",
                        help="Use a different face for each generated image, "
                        "which means that `--num_faces` must be a multiple of the number of valence-arousal pairs. "
                        "If not set, the same face will be used once for each valence-arousal pair.")

    # Parse arguments
    args = parser.parse_args()

    # Check that general arguments are valid
    if args.num_faces <= 0:
        raise ValueError("Number of face images must be a positive integer.")

    # Validate arguments based on mode
    if args.mode == "random":
        # Check if fe_count is a positive integer
        if args.fe_count <= 0:
            raise ValueError("Number of facial expressions must be a positive integer.")
        if not (0.0 <= args.magnitude_coeff <= 1.0):
            raise ValueError(f"Invalid magnitude coefficient '{args.magnitude_coeff}'. Must be between 0.0 and 1.0.")
    elif args.mode == "emotions":
        # Force list for list args
        if isinstance(args.strengths, float):
            args.strengths = [args.strengths]
        if isinstance(args.emotions, str):
            args.emotions = [args.emotions]
        # Check if the specified emotions are valid
        for e in args.emotions:
            if e not in EMOTIONS:
                raise ValueError(f"Invalid emotion '{e}'. Available emotions: {', '.join(EMOTIONS.keys())}")
        # Check if strengths are valid
        for strength in args.strengths:
            if not (0.0 <= strength <= 1.0):
                raise ValueError(f"Invalid strength '{strength}'. Must be between 0.0 and 1.0.")
        # Check if unique mode is valid
        if args.unique:
            if args.num_faces % (len(args.strengths) * len(args.emotions)) != 0:
                raise ValueError(f"Number of faces ({args.num_faces}) must be a multiple of "
                                 f"{len(args.strengths) * len(args.emotions)} for if 'unique' is specified.")
    elif args.mode == "va":
        # Check valences_arousals list
        if isinstance(args.valences_arousals, float):
            args.valences_arousals = [args.valences_arousals]
        if len(args.valences_arousals) % 2 != 0:
            raise ValueError("Valence-arousal pairs must be specified in pairs (valence0 arousal0 valence1 arousal1 ...).")
        # Check if unique mode is valid
        if args.unique:
            if args.num_faces % (len(args.valences_arousals) // 2) != 0:
                raise ValueError(f"Number of faces ({args.num_faces}) must be a multiple of "
                                 f"{len(args.valences_arousals) // 2} for if 'unique' is specified.")
    else:
        raise ValueError(f"Invalid mode '{args.mode}'. Available modes: 'random', 'emotions', 'va'.")

    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # If we need to resume from a previous run, fetch previous state
    next_image_id = 0
    last_image_id = -1
    next_emotion_index = 0
    if args.resume:
        next_image_id, last_image_id = _resume_next_and_last_image_ids(TMP_INPUT_DIR)
        # If no images found, start from scratch
        if next_image_id == -1:
            args.resume = False
            next_image_id = 0
        else:
            # Get next emotion index to resume from
            next_emotion_index = _resume_next_emotion_index(args.output_dir, next_image_id)

    # Emotions based on mode
    emotions = None
    facial_expression_count = 0
    if args.mode == "va":
        # Create Emotion objects from the valence-arousal pairs
        print(f"Using valence-arousal pairs: {args.valences_arousals}")
        emotions = [Emotion(args.valences_arousals[i], args.valences_arousals[i + 1])
                    for i in range(0, len(args.valences_arousals), 2)]
        facial_expression_count = len(emotions)
    elif args.mode == "emotions":
        # Get the emotion data
        emotions = [
            Emotion(EMOTIONS[e].valence * strength, EMOTIONS[e].arousal * strength)
            for e in args.emotions
            for strength in args.strengths
        ]
        facial_expression_count = len(emotions)
    else:   # Random mode
        # Generate random emotions on the fly (later)
        facial_expression_count = args.fe_count

    # Generate random face images if we are not resuming
    if not args.resume:
        # Set up tmp directory for input images
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR)
        os.makedirs(TMP_INPUT_DIR)

    # Determine how many base face images to generate
    # For percentage-based quality control, generate extra images upfront
    num_to_generate = args.num_faces
    if args.quality is not None and args.quality.endswith('%'):
        percentage = float(args.quality[:-1])
        # Generate more images to account for the percentage that will be removed
        # Formula: to keep N images after removing bottom X%, generate N / (1 - X/100)
        num_to_generate = int(np.ceil(args.num_faces / (1 - percentage / 100)))
        print(f"Quality control optimization: generating {num_to_generate} base images to keep best {args.num_faces} after removing bottom {percentage}%")
    
    # Generate random images (format: {image_id:06}.png) and latent vectors (format: {image_id:06}.npy)
    lacking_face_image_count = num_to_generate - last_image_id - 1
    if lacking_face_image_count > 0:
        if args.resume:
            print(f"Resuming... Generating {lacking_face_image_count} new base face identity images...")
        else:
            print(f"Generating {num_to_generate} base face identity images...")
        generate_random_image_set(TMP_INPUT_DIR, count=num_to_generate, force_cpu=args.force_cpu)
    else:
        print(f"Skipping base face image generation.")
    
    # Apply quality control to base face images (if specified)
    if args.quality is not None:
        actual_face_count = _apply_quality_control(
            input_dir=TMP_INPUT_DIR,
            quality_method=args.method,
            quality_requirement=args.quality,
            force_cpu=args.force_cpu,
            tmp_dir=TMP_DIR
        )
        print(f"Quality control complete: {actual_face_count} base face images ready for emotion editing")
        
        # Update next_image_id and args.num_faces based on actual images remaining
        # Get the list of remaining image IDs
        remaining_images = sorted([f for f in os.listdir(TMP_INPUT_DIR) if f.endswith('.png')])
        actual_face_count = len(remaining_images)
        
        if actual_face_count != args.num_faces:
            print(f"Note: {actual_face_count} base images available after quality control (requested {args.num_faces})")
        
        # Reset loop range to process only the remaining images
        next_image_id = 0
        last_image_id = actual_face_count - 1
        args.num_faces = actual_face_count
    
    # Load the EmoStyle framework
    emostyle = EmoStyle(force_cpu=args.force_cpu)

    # Get list of actual base face images to process
    base_face_images = sorted([f for f in os.listdir(TMP_INPUT_DIR) if f.endswith('.png')])
    
    if args.resume and args.quality is None:
        print(f"Resuming from image ID {next_image_id:06} with emotion index {next_emotion_index}...")
        base_face_images = base_face_images[next_image_id:]

    # Generate every emotion for every input image
    total = len(base_face_images) * facial_expression_count
    if next_emotion_index > 0:
        total -= next_emotion_index
    
    if args.mode == "random":
        if args.resume and args.quality is None:
            print("Resuming...", end=" ")
        print(f"Generating {total} facial expressions...")

        for image_file in base_face_images:
            face_id = int(os.path.splitext(image_file)[0])
            # Generate random emotions
            emotions = []
            for _ in range(next_emotion_index, args.fe_count):
                magnitude = random.uniform(0, 1)
                corrected_magnitude = -((-magnitude + 1) * (1 - args.magnitude_coeff) - 1) # FIXME: Right now it is just 's + (1 - s) * m'
                angle = random.uniform(0, 2 * np.pi)
                valence = corrected_magnitude * np.cos(angle)
                arousal = corrected_magnitude * np.sin(angle)
                emotions.append(Emotion(valence, arousal))
            # Reset first emotion index in case we were resuming from previous session
            if next_emotion_index > 0:
                next_emotion_index = 0
            # If emotion list is empty, skip this face
            if not emotions:
                print(f"Skipping face ID {face_id:06} due to no emotions generated.")
                continue
            # Generate image
            input_image_path = f"{TMP_INPUT_DIR}/{face_id:06}.png"
            emostyle.edit_image(input_image_path, emotions, args.output_dir)
            # Remove the image files from the tmp directory
            input_image_path_noext = os.path.splitext(input_image_path)[0]
            os.remove(input_image_path)  # Remove input image
            os.remove(f"{input_image_path_noext}.npy")  # Remove latent vector
    else:
        if args.unique:
            if args.resume and args.quality is None:
                print("Resuming...", end=" ")
            print(f"Generating {total} facial expressions...")

            # num_faces > facial_expression_count
            for idx, image_file in enumerate(base_face_images):
                emotion_index = idx % facial_expression_count
                input_image_path = os.path.join(TMP_INPUT_DIR, image_file)
                emostyle.edit_image(input_image_path, [emotions[emotion_index]], args.output_dir)
                # Remove the image files from the tmp directory
                input_image_path_noext = os.path.splitext(input_image_path)[0]
                os.remove(input_image_path)  # Remove input image
                os.remove(f"{input_image_path_noext}.npy")  # Remove latent vector
        else:
            if args.resume and args.quality is None:
                print(f"Resuming... Generating {total} facial expressions...")
            else:
                print(f"Generating {total} facial expressions...")

            for image_file in base_face_images:
                input_image_path = os.path.join(TMP_INPUT_DIR, image_file)
                emostyle.edit_image(input_image_path, emotions[next_emotion_index:], args.output_dir)
                # Reset first emotion index in case we were resuming from previous session
                if next_emotion_index > 0:
                    next_emotion_index = 0
                # Remove the image files from the tmp directory
                input_image_path_noext = os.path.splitext(input_image_path)[0]
                os.remove(input_image_path) # Remove input image
                os.remove(f"{input_image_path_noext}.npy")  # Remove latent vector
    
    # Clean up temporary directory
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    
    print("Done.")
