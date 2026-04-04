"""
Trail Race Predictor - Runner

Orchestrates training from FIT records and prediction for GPX routes.
Builds on top of the immutable core_rebuild module.
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path so core_rebuild can be imported
_sys_path = Path(__file__).parent.parent
if str(_sys_path) not in sys.path:
    sys.path.insert(0, str(_sys_path))

# Import from the immutable core_rebuild
from core_rebuild.predictor import MLRacePredictor


class RaceRunner:
    """Orchestrates training and prediction using temp/records and temp/routes."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the runner.

        Args:
            base_path: Root directory of the project.
                      Defaults to this file's parent directory.
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent
        self.base_path = Path(base_path)
        self.records_dir = self.base_path / "temp" / "records"
        self.routes_dir = self.base_path / "temp" / "routes"
        self.output_dir = self.base_path / "temp" / "output"
        self.predictor = MLRacePredictor()
        self._is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_from_records(self, effort_factor: float = 1.0) -> bool:
        """Train the model using all FIT files in temp/records/.

        Args:
            effort_factor: Effort level for the model (1.0 = average).

        Returns:
            True if training succeeded, False otherwise.
        """
        fit_files = sorted(
            self.records_dir.glob("*.fit"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )

        if not fit_files:
            print(f"  Error: No FIT files found in {self.records_dir}")
            return False

        print(f"\n  Found {len(fit_files)} FIT records, using all for training")

        file_paths = [str(f) for f in fit_files]
        success = self.predictor.train_from_files(file_paths)

        if success:
            self._is_trained = True
            self._save_training_stats()
            print(f"\n  Training complete! Model ready for prediction.")
        else:
            print(f"\n  Training failed. Check FIT files in {self.records_dir}")

        return success

    def _save_training_stats(self):
        """Save training statistics to temp/output."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stats_path = self.output_dir / "training_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.predictor.training_stats, f, indent=2, ensure_ascii=False)
        print(f"  Training stats saved to {stats_path}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_all_routes(
        self,
        effort_factor: float = 1.0,
        save_results: bool = True,
    ) -> dict:
        """Run predictions for all GPX routes in temp/routes/.

        Args:
            effort_factor: Effort level (1.0 = average, 1.1-1.2 = race effort).
            save_results: Whether to save results to temp/output/.

        Returns:
            Dict mapping route filename to prediction result.
        """
        if not self._is_trained:
            print("  Warning: Model not trained. Training now...")
            if not self.train_from_records(effort_factor):
                return {}

        gpx_files = list(self.routes_dir.glob("*.gpx"))

        if not gpx_files:
            print(f"  Error: No GPX files found in {self.routes_dir}")
            return {}

        print(f"\n  Found {len(gpx_files)} GPX routes, running predictions...")

        results = {}
        for gpx_path in gpx_files:
            print(f"\n  Predicting: {gpx_path.name}")
            try:
                result = self.predictor.predict_race(str(gpx_path), effort_factor=effort_factor)
                results[gpx_path.name] = result
                self._print_prediction_summary(result)
            except Exception as e:
                print(f"  Error predicting {gpx_path.name}: {e}")
                results[gpx_path.name] = {"error": str(e)}

        if save_results:
            self._save_prediction_results(results)

        return results

    def predict_route(self, gpx_filename: str, effort_factor: float = 1.0) -> dict:
        """Predict race time for a specific GPX route.

        Args:
            gpx_filename: Name of the GPX file (e.g., "race.gpx").
            effort_factor: Effort level (1.0 = average).

        Returns:
            Prediction result dict.
        """
        if not self._is_trained:
            if not self.train_from_records(effort_factor):
                return {"error": "Training failed"}

        gpx_path = self.routes_dir / gpx_filename
        if not gpx_path.exists():
            return {"error": f"Route not found: {gpx_filename}"}

        return self.predictor.predict_race(str(gpx_path), effort_factor=effort_factor)

    def _print_prediction_summary(self, result: dict):
        """Print a concise summary of the prediction."""
        print(f"    Distance:     {result['total_distance_km']} km")
        print(f"    Total Ascent: {result['route_info']['total_elevation_gain_m']} m")
        print(f"    Predicted:    {result['predicted_time_hm']} "
              f"({result['predicted_time_min']:.0f} min)")
        print(f"    Avg Pace:     {result['predicted_pace_min_km']} min/km")
        print(f"    Effort:       {result['effort_factor']}x")

    def _save_prediction_results(self, results: dict):
        """Save all prediction results to temp/output."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save combined results
        combined_path = self.output_dir / "all_prediction_results.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {combined_path}")

        # Save individual results
        for name, result in results.items():
            if "error" not in result:
                safe_name = Path(name).stem + "_result.json"
                out_path = self.output_dir / safe_name
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        effort_factor: float = 1.0,
    ) -> dict:
        """Run the complete train-then-predict pipeline.

        Args:
            effort_factor: Effort level for prediction.

        Returns:
            Dict of prediction results.
        """
        print("=" * 60)
        print("Trail Race Predictor - Full Pipeline")
        print("=" * 60)

        print("\n--- Step 1: Training from FIT records ---")
        if not self.train_from_records(effort_factor):
            return {}

        print("\n--- Step 2: Predicting race times for GPX routes ---")
        return self.predict_all_routes(effort_factor, save_results=True)


if __name__ == "__main__":
    runner = RaceRunner()
    results = runner.run_full_pipeline(effort_factor=1.0)
    print("\n" + "=" * 60)
    print(f"Pipeline complete! {len(results)} routes processed.")
    print("=" * 60)
