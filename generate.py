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
    # Parse command line arguments
    # generate.py [-n num_mult] [-o output_dir] [-e emotions...|-v valences_arousals...] [-s strengths...] [-r] [-c]
    parser = argparse.ArgumentParser(description="Generate images with specific facial expressions.")
    parser.add_argument("-n", "--num_mult", type=int, default=1, help="Number of images multiplier (default: 1) "
                        "The actual number of images to generate is equal to "
                        "`num_mult * [len(emotions)|len(valences_arousals)/2] * len(strengths)`")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Directory to save generated images")
    parser.add_argument("-e", "--emotions", type=str, default="random", nargs="+",
                        help="List of emotions to apply (default: random). Available emotions: " + ", ".join(EMOTIONS.keys()))
    parser.add_argument("-v", "--valences_arousals", type=float, default=None, nargs="+",
                        help="List of valence-arousal pairs (valence0 arousal0 valence1 arousal1 ...) to apply "
                             "(default: None, use --emotions instead). If specified, overrides --emotions.")
    parser.add_argument("-s", "--strengths", type=float, default=1.0, nargs="+",
                        help="List of strengths (from 0.0 to 1.0) for each emotion (default: 1.0). "
                             "Requires --emotions. Undefined with --valences_arousals.")
    parser.add_argument("-r", "--repeat", action="store_true", 
                        help="Each input image is repeatedly used until it has been edited with every emotion and strength combination")
    parser.add_argument("-c", "--force_cpu", action="store_true", help="Force to run on CPU, otherwise GPU will be used if available")
    args = parser.parse_args()

    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Check that num_mult is valid
    if args.num_mult <= 0:
        raise ValueError("Number of images must be a positive integer.")

    # Check what type of parameters are specified
    are_emotions_random = args.emotions == ["random"]
    is_va_specified = args.valences_arousals is not None

    # Force list for list args
    if isinstance(args.strengths, float):
        args.strengths = [args.strengths]
    if isinstance(args.emotions, str):
        args.emotions = [args.emotions]

    emotions = None
    # If valences_arousals are specified, check if they are valid
    if args.valences_arousals is not None:
        if len(args.valences_arousals) % 2 != 0:
            raise ValueError("Valence-arousal pairs must be specified in pairs (valence0 arousal0 valence1 arousal1 ...).")
        # Create Emotion objects from the valence-arousal pairs
        print(f"Using valence-arousal pairs: {args.valences_arousals}")
        emotions = [Emotion(args.valences_arousals[i], args.valences_arousals[i + 1])
                    for i in range(0, len(args.valences_arousals), 2)]
        print(f"Parsed emotions: {emotions}")
        # Strengths are not used in this case, so set them to 1.0
        args.strengths = [1.0]
    # If emotion is "random", generate a random valence-arousal pair
    elif args.emotions == ["random"]:
        # Generate N random emotions
        # Ensure that the magnitude of the valence-arousal vector is less than or equal to 1
        emotions = []
        for _ in range(args.num_mult):
            angle = random.uniform(0, 2 * np.pi)
            magnitude = random.uniform(0, 1)
            valence = magnitude * np.cos(angle)
            arousal = magnitude * np.sin(angle)
            emotions.append(Emotion(valence, arousal))
        # Since we increased the number of emotions, we need to adjust the number of images to generate
        args.num_mult = 1
        # Strengths are not used in this case, so set them to 1.0
        args.strengths = [1.0]
    # If emotions are specified, check if they are valid
    else:
        for e in args.emotions:
            if e not in EMOTIONS:
                raise ValueError(f"Invalid emotion '{e}'. Available emotions: {', '.join(EMOTIONS.keys())}")
        # If valid, get the emotion data
        emotions = [EMOTIONS[e] for e in args.emotions]

    # Check if strengths are valid
    for strength in args.strengths:
        if not (0.0 <= strength <= 1.0):
            raise ValueError(f"Invalid strength '{strength}'. Must be between 0.0 and 1.0.")
    
    # Set up tmp directory for input images
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_INPUT_DIR)

    # Generate random images (format: {image_id:06}.png) and latent vectors (format: {image_id:06}.npy)
    random_image_count = args.num_mult if args.repeat else args.num_mult * len(emotions) * len(args.strengths)
    print(f"Generating {random_image_count} new face identity images...")
    generate_random_image_set(TMP_INPUT_DIR, count=random_image_count, force_cpu=args.force_cpu)

    # Load the EmoStyle framework
    emostyle = EmoStyle(force_cpu=args.force_cpu)
    
    # Combine emotions and strengths lists
    weighted_emotions = [
        Emotion(emotion.valence * strength, emotion.arousal * strength)
        for emotion in emotions
        for strength in args.strengths
    ]

    # If "repeat", generate every emotion for every input image
    if args.repeat:
        # Edit the images with every emotion and strength combination
        for i in range(args.num_mult):
            input_image_path = f"{TMP_INPUT_DIR}/{i:06}.png"
            emostyle.edit_image(input_image_path, weighted_emotions, args.output_dir)
    # If not "repeat", use a different input image for each edit
    else:
        for i in range(args.num_mult):
            for j in range(len(weighted_emotions)):
                image_index = i * len(weighted_emotions) + j
                input_image_path = f"{TMP_INPUT_DIR}/{image_index:06}.png"
                # Edit the image with the emotion and save it
                emostyle.edit_image(input_image_path, [weighted_emotions[j]], args.output_dir)

    # Clean up temporary directory
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    
    print(f"Generated {args.num_mult * len(emotions) * len(args.strengths)} images in '{args.output_dir}' directory.")
    print("Done.")
