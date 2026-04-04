"""
Trail Race Predictor - Runner

Lightweight orchestration layer on top of core_rebuild.
Takes FIT records from temp/records/ for training,
and GPX routes from temp/routes/ for prediction.

Usage::

    from runner import RaceRunner

    runner = RaceRunner()
    runner.train_from_records()           # uses temp/records/*.fit
    runner.predict_all_routes()           # outputs to temp/output/
"""

from .runner import RaceRunner

__all__ = ["RaceRunner"]
