import unittest

from analyze_exact_quality import summarize_exact


class ExactQualityAnalysisTests(unittest.TestCase):
    def test_quality_and_pairing_use_sample_medians(self):
        payload = {
            "metadata": {
                "configuration": {
                    "datasets": [{"name": "real-rootr-n8", "roots": ["r"]}],
                    "methods": ["star", "cascade"],
                }
            },
            "records": [
                {
                    "dataset": "real-rootr-n8",
                    "method": "star",
                    "cost": 12.0,
                    "wall_time_s": 2.0,
                    "validated": True,
                },
                {
                    "dataset": "real-rootr-n8",
                    "method": "star",
                    "cost": 14.0,
                    "wall_time_s": 4.0,
                    "validated": True,
                },
                {
                    "dataset": "real-rootr-n8",
                    "method": "cascade",
                    "cost": 10.0,
                    "wall_time_s": 3.0,
                    "validated": True,
                },
                {
                    "dataset": "real-rootr-n8",
                    "method": "cascade",
                    "cost": 10.0,
                    "wall_time_s": 5.0,
                    "validated": True,
                },
            ],
        }
        optima = [{"name": "real-rootr-n8", "cost": 10.0}]
        rows, summaries, comparison = summarize_exact(payload, optima)
        by_method = {row["method"]: row for row in rows}
        self.assertEqual(by_method["star"]["cost"], 13.0)
        self.assertEqual(by_method["cascade"]["cost_ratio"], 1.0)
        self.assertEqual(by_method["cascade"]["median_wall_s"], 4.0)
        self.assertEqual(summaries[1]["optimal_hits"], 1)
        self.assertEqual(comparison["wins"], 1)
        self.assertEqual(comparison["wins_reaching_optimum"], 1)

    def test_size_guard_is_reported_as_unavailable(self):
        payload = {
            "metadata": {
                "configuration": {
                    "datasets": [{"name": "real-rootr-n8", "roots": ["r"]}],
                    "methods": ["huang2"],
                }
            },
            "records": [
                {
                    "dataset": "real-rootr-n8",
                    "method": "huang2",
                    "status": "size-guard",
                    "wall_time_s": 0.1,
                    "validated": False,
                }
            ],
        }
        rows, summaries, comparison = summarize_exact(
            payload, [{"name": "real-rootr-n8", "cost": 10.0}]
        )
        self.assertFalse(rows[0]["available"])
        self.assertEqual(summaries[0]["available_samples"], 0)
        self.assertEqual(summaries[0]["unavailable_statuses"], ["size-guard"])
        self.assertEqual(comparison["paired_samples"], 0)


if __name__ == "__main__":
    unittest.main()
