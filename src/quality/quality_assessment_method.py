class QualityAssessmentMethod:

    def calculate_quality_scores(self, image_dir: str) -> dict:
        """
        Calculate quality scores for each image in the provided directory path.
        Returns a dictionary mapping image paths to their quality scores.
        """
        raise NotImplementedError("Subclasses should implement this method.")
