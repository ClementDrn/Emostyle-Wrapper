import numpy as np
from .quality.quality_assessment_method import QualityAssessmentMethod


def calculate_threshold(quality_scores: dict, percentage: float = 50.0) -> float:
    """
    Calculate the quality threshold based on the specified percentage of the quality scores.
    The higher the percentage, the higher the threshold.
    """
    if not quality_scores:
        raise ValueError("Quality scores dictionary is empty.")
    
    scores = list(quality_scores.values())
    threshold = np.percentile(scores, percentage)
    return threshold


def filter_images_by_quality(method: QualityAssessmentMethod, image_dir: str, quality_threshold: float, quality_threshold_max: float = None) -> list:
    """
    Filter images based on their quality scores calculated based on the specified method.
    Images with a quality score below the specified threshold are excluded.
    
    Returns a list of tuples containing the image path and its quality score.
    """
    # Get dictionary of quality scores
    quality_scores = method.calculate_quality_scores(image_dir)
    filtered_images = []
    for image_path, quality_score in quality_scores.items():
        if quality_score >= quality_threshold and (quality_threshold_max is None or quality_score <= quality_threshold_max):
            filtered_images.append((image_path, quality_score))
            
    # Sort images by quality score in descending order
    filtered_images.sort(key=lambda x: x[1], reverse=True)
    
    return filtered_images
