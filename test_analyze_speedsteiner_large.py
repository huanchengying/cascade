import unittest

from analyze_speedsteiner_large import summarize


class SpeedSteinerLargeAnalysisTests(unittest.TestCase):
    def test_available_and_guarded_units_are_retained(self):
        speed = {
            "metadata": {
                "config": {
                    "datasets": [
                        {"name": "a", "roots": ["r1", "r2"]},
                    ]
                }
            },
            "records": [
                {
                    "dataset": "a",
                    "root": "r1",
                    "status": "feasible",
                    "validated": True,
                    "cost": 8,
                    "wall_time_s": 2,
                    "peak_rss_mib": 10,
                },
                {
                    "dataset": "a",
                    "root": "r2",
                    "status": "size-guard",
                    "validated": False,
                    "wall_time_s": 1,
                    "peak_rss_mib": 5,
                },
            ],
        }
        cascade = {
            "records": [
                {"dataset": "a", "root": "r1", "method": "cascade", "validated": True, "cost": 10},
                {"dataset": "a", "root": "r2", "method": "cascade", "validated": True, "cost": 7},
            ]
        }
        rows, summaries = summarize(speed, cascade)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["outcome"], "win")
        self.assertFalse(rows[1]["available"])
        self.assertEqual(summaries[-1]["available_units"], 1)
        self.assertEqual(summaries[-1]["statuses"]["size-guard"], 1)


if __name__ == "__main__":
    unittest.main()
