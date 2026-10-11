import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np

from tests import helpers
from tests.helpers import T0, alert, beacon_events, bruteforce_events, ev
from sentinelcrew.ml import (
    FEATURE_NAMES,
    IsolationForest,
    average_path_length,
    beacon_regularity,
    extract_features,
    generate_baseline,
    to_vector,
)
from sentinelcrew.ml.features import is_off_hours, is_suspicious_process
from sentinelcrew.ml.train import main as train_main


class TestIsolationForest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.X = rng.normal(0, 1, size=(400, 3))
        cls.model = IsolationForest(n_estimators=60, max_samples=128, random_state=1).fit(cls.X)

    def test_c_n(self):
        self.assertEqual(average_path_length(1), 0.0)
        self.assertEqual(average_path_length(2), 1.0)
        self.assertAlmostEqual(average_path_length(256), 10.2448, places=3)

    def test_outlier_scores_higher(self):
        normal, outlier = self.model.score_samples([[0, 0, 0], [8, -8, 8]])
        self.assertGreater(outlier, 0.6)
        self.assertLess(normal, 0.5)
        self.assertGreater(outlier, normal)

    def test_deterministic(self):
        other = IsolationForest(n_estimators=60, max_samples=128, random_state=1).fit(self.X)
        np.testing.assert_allclose(other.score_samples(self.X[:5]), self.model.score_samples(self.X[:5]))

    def test_explain_points_at_anomalous_feature(self):
        top = self.model.explain([0, 9, 0], top_k=1)
        self.assertEqual(top[0][0], "f1")

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m" / "model.json"
            self.model.save(path)
            loaded = IsolationForest.load(path)
        np.testing.assert_allclose(loaded.score_samples(self.X[:5]), self.model.score_samples(self.X[:5]))

    def test_errors(self):
        with self.assertRaises(ValueError):
            IsolationForest(n_estimators=0)
        with self.assertRaises(ValueError):
            IsolationForest(max_samples=1)
        with self.assertRaises(ValueError):
            IsolationForest().fit(np.zeros((1, 2)))
        with self.assertRaises(RuntimeError):
            IsolationForest().score_samples([[1, 2]])
        with self.assertRaises(ValueError):
            self.model.score_samples([[1, 2]])

    def test_constant_data(self):
        model = IsolationForest(n_estimators=5, random_state=0).fit(np.ones((10, 2)))
        self.assertEqual(model.score_samples([[1, 1]]).shape, (1,))


class TestFeatures(unittest.TestCase):
    def test_beacon_regularity(self):
        periodic = [T0 + timedelta(seconds=60 * i) for i in range(10)]
        self.assertGreater(beacon_regularity(periodic), 0.95)
        random_ts = [T0 + timedelta(seconds=s) for s in (0, 3, 100, 104, 900, 2000)]
        self.assertLess(beacon_regularity(random_ts), 0.3)
        self.assertEqual(beacon_regularity(periodic[:3]), 0.0)
        self.assertEqual(beacon_regularity([T0] * 5), 0.0)

    def test_helpers(self):
        self.assertTrue(is_off_hours(T0))  # 02:00
        self.assertTrue(is_off_hours(T0.replace(day=10, hour=12)))  # Saturday
        self.assertFalse(is_off_hours(T0.replace(hour=12)))
        self.assertTrue(is_suspicious_process("powershell.exe -nop -w hidden -enc AAA"))
        self.assertFalse(is_suspicious_process("excel.exe"))
        self.assertFalse(is_suspicious_process(None))

    def test_bruteforce_features(self):
        f = extract_features(alert(), bruteforce_events(30), 0.75)
        self.assertEqual(f["failed_auth_count"], 30)
        self.assertEqual(f["success_after_failures"], 1.0)
        self.assertEqual(f["external_src"], 1.0)
        self.assertEqual(f["asset_criticality"], 0.75)
        self.assertEqual(f["severity_hint"], 0.75)

    def test_beacon_and_exfil_features(self):
        events = beacon_events(10) + [ev(900, 20, event_type="network", dst_ip="45.137.21.9", bytes_out=80_000_000)]
        events.append(ev(901, 1, event_type="process", process="rundll32.exe x.dll,Start"))
        f = extract_features(alert(user="r.iyer", src_ip=None), events)
        self.assertGreater(f["beacon_regularity"], 0.9)
        self.assertGreater(f["mb_out"], 80)
        self.assertEqual(f["suspicious_process"], 1.0)
        self.assertEqual(f["external_src"], 0.0)

    def test_few_failures_do_not_count_as_compromise(self):
        f = extract_features(alert(src_ip="10.0.1.53"), bruteforce_events(2, src="10.0.1.53"))
        self.assertEqual(f["success_after_failures"], 0.0)

    def test_vector_order(self):
        f = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
        self.assertEqual(to_vector(f), list(range(len(FEATURE_NAMES))))
        self.assertEqual(to_vector({}), [0.0] * len(FEATURE_NAMES))


class TestTraining(unittest.TestCase):
    def test_baseline_shape(self):
        X = generate_baseline(100)
        self.assertEqual(X.shape, (100, len(FEATURE_NAMES)))
        np.testing.assert_array_equal(X, generate_baseline(100))

    def test_attack_vector_is_anomalous(self):
        model = helpers.small_model()
        normal = to_vector(dict(zip(FEATURE_NAMES, [0, 1, 0, 0, 0, 2, 2, 0.1, 0, 0.5, 0.5])))
        attack = to_vector(dict(zip(FEATURE_NAMES, [46, 3, 1, 1, 1, 0, 0, 0, 0, 0.75, 0.75])))
        s_normal, s_attack = model.score_samples([normal, attack])
        self.assertGreater(s_attack, s_normal + 0.1)

    def test_train_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = train_main(["--samples", "100", "--trees", "5", "--out", str(Path(tmp) / "m.json")])
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
