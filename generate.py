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


# Internal constants
EMO_EMBED = 64
STG_EMBED = 512
INPUT_SIZE = 1024
ITER_NUM = 500
# Temporary directories
TMP_DIR = "tmp"
TMP_INPUT_DIR = "tmp/input"
# Model weights paths
STYLEGAN2_WEIGHTS_PATH = "pretrained/ffhq.pkl"
EMOSTYLE_WEIGHTS_PATH = "pretrained/emo_mapping_wplus_2.pt"
EMONET_WEIGHTS_PATH = "pretrained/emonet_8.pth"


def generate_random_image_set(
    output_dir: str,
    count: int = 1,
    force_cpu: bool = False
) -> None:
    """
    Generate a set of random images and save them to the specified output directory.
    """
    # Define generation batch size
    batch_size = 8
    if batch_size > count:
        batch_size = count  # Ensure batch size is not more than the number of samples
    
    # Generate random images using StyleGAN2
    stylegan2(
        checkpoint_path="pretrained/ffhq.pkl",
        outdir=output_dir,
        n_samples=count,
        batch_size=batch_size,
        truncation_psi=0.7,
        on_cpu=force_cpu
    )


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



# --- Main --------------------------------------------------------------------
if __name__ == "__main__":   
    # generate.py [-n num_faces] [-o output_dir] [-c] random [fe_count] [magnitude_coeff]
    # generate.py [-n num_faces] [-o output_dir] [-c] emotions <emotion1 [emotion2 ...]> [-s strengths...] 
    # generate.py [-n num_faces] [-o output_dir] [-c] va <valence0 arousal0 [valence1 arousal1 ...]>
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate images with specific facial expressions.")
    parser.add_argument("-n", "--num_faces", type=int, default=1,
                        help="Number of unique face identities to generate (default: 1).")
    parser.add_argument("-o", "--output_dir", type=str, default="output",
                        help="Directory to save the generated images (default: 'output').")
    parser.add_argument("-c", "--force_cpu", action="store_true",
                        help="Force the program to run on CPU, even if a GPU is available.")

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

    # Set up tmp directory for input images
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_INPUT_DIR)

    # Generate random images (format: {image_id:06}.png) and latent vectors (format: {image_id:06}.npy)
    print(f"Generating {args.num_faces} new face identity images...")
    generate_random_image_set(TMP_INPUT_DIR, count=args.num_faces, force_cpu=args.force_cpu)

    # Load the EmoStyle framework
    emostyle = EmoStyle(force_cpu=args.force_cpu)

    # Generate every emotion for every input image
    print(f"Generating {facial_expression_count} facial expressions...")
    if args.mode == "random":
        for face_id in range(args.num_faces):
            # Generate random emotions
            emotions = []
            for _ in range(args.fe_count):
                magnitude = random.uniform(0, 1)
                corrected_magnitude = -((-magnitude + 1) * (1 - args.magnitude_coeff) - 1)
                angle = random.uniform(0, 2 * np.pi)
                valence = corrected_magnitude * np.cos(angle)
                arousal = corrected_magnitude * np.sin(angle)
                emotions.append(Emotion(valence, arousal))
            # Generate image
            input_image_path = f"{TMP_INPUT_DIR}/{face_id:06}.png"
            emostyle.edit_image(input_image_path, emotions, args.output_dir)
    else:
        if args.unique:
            # num_faces > facial_expression_count
            for face_id in range(args.num_faces):
                emotion_index = face_id % facial_expression_count
                input_image_path = f"{TMP_INPUT_DIR}/{face_id:06}.png"
                emostyle.edit_image(input_image_path, [emotions[emotion_index]], args.output_dir)
        else:
            for face_id in range(args.num_faces):
                input_image_path = f"{TMP_INPUT_DIR}/{face_id:06}.png"
                emostyle.edit_image(input_image_path, emotions, args.output_dir)

    # Clean up temporary directory
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    
    print("Done.")
