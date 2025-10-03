from .quality_assessment_method import QualityAssessmentMethod
import os
import subprocess
import sys


class GIQA(QualityAssessmentMethod):

    def __init__(self, weights_path: str, tmp_dir: str, gpu_id: int = 0):
        self.weights_path = weights_path
        self.tmp_dir = tmp_dir
        self.gpu_id = gpu_id    # -1 means CPU

    def calculate_quality_scores(self, image_dir: str) -> dict:
        """
        Calculate quality scores for each image in the provided directory path.
        Returns a dictionary mapping image paths to their quality scores.
        
        GIQA outputs scores in multi-line format:
            score of IMAGE_FILE is:
            SCORE_VALUE
        """
        # Call the GIQA script with the appropriate arguments
        output_file = os.path.join(self.tmp_dir, 'giqa_scores.txt')
        command = [
            sys.executable, 'vendor/GIQA/gmm_score.py',
            image_dir,
            '--gmm_path', self.weights_path,
            '--output_file', output_file,
        ]
        if self.gpu_id >= 0:
            command.extend(['--gpu', str(self.gpu_id)])
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Check if output file was created
        if not os.path.exists(output_file):
            print(f"Warning: GIQA output file not created at {output_file}")
            return {}

        # Read the output file and parse the scores in GIQA's multi-line format
        quality_scores = {}
        with open(output_file, 'r') as f:
            lines = f.readlines()
            
        # Parse GIQA's format: "score of FILENAME is:\nSCORE\n"
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('score of ') and ' is:' in line:
                # Extract filename from "score of FILENAME is:"
                try:
                    filename = line.split('score of ')[1].split(' is:')[0]
                    # Next line contains the score
                    if i + 1 < len(lines):
                        score_str = lines[i + 1].strip()
                        score = float(score_str)
                        # Build full path by joining image_dir with filename
                        image_path = os.path.join(image_dir, filename)
                        quality_scores[image_path] = score
                        i += 2  # Skip to next score block
                    else:
                        i += 1
                except (ValueError, IndexError) as e:
                    # Skip invalid entries
                    i += 1
            else:
                i += 1
        
        if not quality_scores:
            print(f"Warning: No scores parsed from GIQA output file. File has {len(lines)} lines.")
            if len(lines) > 0:
                print(f"Sample content: {lines[:min(5, len(lines))]}")
        else:
            print(f"GIQA: Successfully parsed {len(quality_scores)} image scores")
        
        return quality_scores
