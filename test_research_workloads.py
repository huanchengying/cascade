import unittest

from prepare_research_workloads import transform_payload


class WorkloadTests(unittest.TestCase):
    def test_trust_filter_and_monotone_mapping(self):
        payload = {"vertices": ["r", "a"], "edges": [
            {"id": str(rating), "u": "r", "v": "a", "start": 0, "arrival": 1, "weight": 1, "rating": rating}
            for rating in (-5, 1, 5)
        ]}
        positive = transform_payload(payload, "trust-positive")
        self.assertEqual([edge["rating"] for edge in positive["edges"]], [1, 5])
        exponential = transform_payload(payload, "trust-exponential")
        weights = [edge["weight"] for edge in exponential["edges"]]
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        self.assertEqual(payload["edges"][0]["weight"], 1)

    def test_recorded_duration_and_window(self):
        payload = {"vertices": ["r", "a"], "edges": [
            {"id": "trip", "u": "r", "v": "a", "start": 10, "arrival": 15, "weight": 30, "duration_s": 5}
        ]}
        self.assertEqual(transform_payload(payload, "recorded-duration")["edges"][0]["weight"], 5)
        self.assertEqual(transform_payload(payload, "recorded-duration", (0, 14))["edges"], [])
        payload["edges"][0]["duration_s"] = 6
        with self.assertRaisesRegex(ValueError, "consistent"):
            transform_payload(payload, "recorded-duration")


if __name__ == "__main__":
    unittest.main()
