"""
Trail Race Predictor Package

Quick start::

    from predictor import MLRacePredictor

    model = MLRacePredictor()
    model.train_from_files(["activity1.fit", "activity2.fit"])
    result = model.predict_race("race.gpx", effort_factor=1.0)
    print(result["predicted_time_hm"])
"""

from .predictor import MLRacePredictor
from .features  import SegmentFeatures, MOVING_THRESHOLD_KMH, GPS_ERROR_THRESHOLD_KMH
from .model     import LightGBMPredictor
from .extractor import FeatureExtractor
from .gpx_parser import GPXRouteParser

__all__ = [
    "MLRacePredictor",
    "SegmentFeatures",
    "LightGBMPredictor",
    "FeatureExtractor",
    "GPXRouteParser",
    "MOVING_THRESHOLD_KMH",
    "GPS_ERROR_THRESHOLD_KMH",
]
