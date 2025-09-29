from quality_assessment_method import QualityAssessmentMethod
import os
import subprocess


class GIQA(QualityAssessmentMethod):

    def __init__(self, weights_path: str, tmp_dir: str, gpu_id: int = 0):
        self.weights_path = weights_path
        self.tmp_dir = tmp_dir
        self.gpu_id = gpu_id    # -1 means CPU

    def calculate_quality_scores(self, image_dir: str) -> dict:
        """
        Calculate quality scores for each image in the provided directory path.
        Returns a dictionary mapping image paths to their quality scores.
        """
        # Call the GIQA script with the appropriate arguments
        output_file = os.path.join(self.tmp_dir, 'giqa_scores.txt')
        command = [
            'python', 'gmm_score.py',
            image_dir,
            '--gmm_path', self.weights_path,
            '--output_file', output_file
            * ([] if self.gpu_id < 0 else ['--gpu', str(self.gpu_id)])  # CPU/GPU support
        ]
        subprocess.run(command, check=True)

        # Read the output file and parse the scores
        quality_scores = {}
        with open(output_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_path, score_str = parts
                    try:
                        score = float(score_str)
                        quality_scores[image_path] = score
                    except ValueError:
                        continue  # Skip lines with invalid score format
        
        return quality_scores
