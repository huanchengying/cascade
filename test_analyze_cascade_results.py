import unittest

from analyze_cascade_results import summarize_payload


class CascadeAnalysisTests(unittest.TestCase):
    def test_repeats_are_collapsed_before_pairing(self):
        payload = {
            "metadata": {},
            "records": [
                {
                    "dataset": "a",
                    "root": "r",
                    "method": "star",
                    "cost": 12.0,
                    "wall_time_s": 2.0,
                    "validated": True,
                },
                {
                    "dataset": "a",
                    "root": "r",
                    "method": "star",
                    "cost": 10.0,
                    "wall_time_s": 1.0,
                    "validated": True,
                },
                {
                    "dataset": "a",
                    "root": "r",
                    "method": "cascade",
                    "cost": 9.0,
                    "wall_time_s": 3.0,
                    "validated": True,
                    "relaxed_deadline_vertices": 2,
                },
                {
                    "dataset": "a",
                    "root": "r",
                    "method": "cascade",
                    "cost": 11.0,
                    "wall_time_s": 4.0,
                    "validated": True,
                    "relaxed_deadline_vertices": 0,
                },
            ],
        }
        pairs, summaries = summarize_payload(payload)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["baseline_cost"], 11.0)
        self.assertEqual(pairs[0]["treatment_cost"], 10.0)
        self.assertEqual(pairs[0]["outcome"], "win")
        self.assertEqual(pairs[0]["relaxed_deadline_vertices"], 1.0)
        self.assertEqual(summaries[-1]["wins"], 1)

    def test_tolerance_classifies_numerical_ties(self):
        payload = {
            "metadata": {},
            "records": [
                {
                    "dataset": "a",
                    "root": "r",
                    "method": "star",
                    "cost": 100.0,
                    "wall_time_s": 1.0,
                    "validated": True,
                },
                {
                    "dataset": "a",
                    "root": "r",
                    "method": "cascade",
                    "cost": 100.0 + 1e-11,
                    "wall_time_s": 1.0,
                    "validated": True,
                },
            ],
        }
        pairs, _ = summarize_payload(payload)
        self.assertEqual(pairs[0]["outcome"], "tie")

    def test_round_cap_is_not_convergence(self):
        records = []
        for root, rounds, moves in (("capped", 32, 32), ("converged", 23, 22)):
            records.extend(
                [
                    {
                        "dataset": "a", "root": root, "method": "star",
                        "cost": 10.0, "wall_time_s": 1.0, "validated": True,
                        "rounds": rounds, "accepted_moves": moves,
                    },
                    {
                        "dataset": "a", "root": root, "method": "cascade",
                        "cost": 9.0, "wall_time_s": 2.0, "validated": True,
                    },
                ]
            )
        pairs, summaries = summarize_payload({"metadata": {}, "records": records})
        self.assertFalse(pairs[0]["baseline_converged"])
        self.assertTrue(pairs[1]["baseline_converged"])
        self.assertEqual(summaries[-1]["verified_converged_baselines"], 1)
        self.assertEqual(summaries[-1]["wins_after_verified_convergence"], 1)


if __name__ == "__main__":
    unittest.main()
