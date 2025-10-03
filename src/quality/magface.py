from .quality_assessment_method import QualityAssessmentMethod
import os
from PIL import Image
import numpy as np
import csv
import subprocess
import sys

# Constants
MAGFACE_IMAGE_DIMENSION = 112  # Dimension to resize images for filtering


class MagFace(QualityAssessmentMethod):

    def __init__(self, magface_executable_path: str, checkpoint_path: str, tmp_dir: str, gpu_id: int = 0):
        self.magface_executable_path = magface_executable_path
        self.checkpoint_path = checkpoint_path
        self.tmp_dir = tmp_dir
        self.gpu_id = gpu_id    # -1 means CPU
        if self.gpu_id > 0:
            print("\033[93m" + "WARNING: MagFace does not support GPU selection. GPU 0 will be used." + "\033[0m")


    def calculate_quality_scores(self, image_dir: str) -> dict:
        """
        Calculate quality scores for each image in the provided directory path.
        Returns a dictionary mapping image paths to their quality scores.
        """
        # Resize images for MagFace assessment
        output_dir = os.path.join(self.tmp_dir, 'resized_images')
        img_list_path, path_mapping = self._setup_images_for_filtering(image_dir, output_dir)

        # Call the MagFace executable with the appropriate arguments and generate features
        feature_list_path = os.path.join(self.tmp_dir, 'magface_features.csv')
        command = [
            sys.executable, self.magface_executable_path,
            '--inf_list', img_list_path,
            '--feat_list', feature_list_path,
            '--resume', self.checkpoint_path,
        ]
        if self.gpu_id < 0:
            command.append('--cpu-mode')
        subprocess.run(command, check=True)

        # Calculate quality scores from the generated feature list
        quality_scores = self._calculate_quality_scores_from_csv(feature_list_path, path_mapping)
        return quality_scores
    

    def _setup_images_for_filtering(self, input_directory, output_directory):
        """
        Set up images for filtering by resizing them to a specific dimension and listing them in a text file.
        
        Images from the input directory are saved to the output directory after being resized.
        A text file named 'img.list' is created in the output directory, containing the absolute paths of the resized images.

        Returns the path to the text file listing the resized images and a mapping from resized paths to original paths.
        """
        # Ensure the output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Create a text file to list the resized images and a mapping dictionary
        img_list_file = os.path.join(output_directory, 'img.list')
        path_mapping = {}
        with open(img_list_file, 'w') as f:
            # Iterate through all image files in the input directory to resize them
            for image_file in os.listdir(input_directory):
                image_path = os.path.join(input_directory, image_file)
                if os.path.isfile(image_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Add dimension to the output file name
                    output_file_name = f"{os.path.splitext(image_file)[0]}_x{MAGFACE_IMAGE_DIMENSION}{os.path.splitext(image_file)[1]}"
                    output_file = os.path.join(output_directory, output_file_name)
                    # Resize and save the image
                    self._resize_image(image_path, output_file, MAGFACE_IMAGE_DIMENSION)
                    # Write the absolute path to the text file
                    f.write(f"{output_file}\n")
                    # Map resized image path to original image path
                    path_mapping[output_file] = image_path

        return img_list_file, path_mapping

    def _resize_image(self, image_input_path, image_output_path, image_dimension):
        """
        Resize the image to the specified dimension and save it to the specified output path.
        """
        image = Image.open(image_input_path)
        image = image.resize((image_dimension, image_dimension), Image.Resampling.LANCZOS)
        image.save(image_output_path, format='PNG', quality=95)

    def _calculate_quality_score(self, feature_vector):
        """
        Calculate the quality score based on the feature vector.
        The quality score is the magnitude of the feature vector.
        """
        return np.linalg.norm(feature_vector)


    def _calculate_quality_scores_from_csv(self, feature_list_path, path_mapping):
        """
        Calculate quality scores for each feature vector in the provided CSV file.
        Returns a dictionary mapping original image paths to their quality scores.
        """
        quality_scores = {}
        with open(feature_list_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Ensure there are features to process
                if len(row) <= 1:
                    print(f"Warning: Skipping row with insufficient data: {row}")
                    continue

                # Skip empty cells and convert the rest of the row to floats
                feature_vector = np.array([float(value) for value in row[1:] if value.strip()]).astype(np.float32)
                # Calculate the quality score
                quality_score = self._calculate_quality_score(feature_vector)
                # Map resized image path back to original image path
                resized_path = row[0]
                original_path = path_mapping.get(resized_path, resized_path)
                quality_scores[original_path] = quality_score

        return quality_scores
