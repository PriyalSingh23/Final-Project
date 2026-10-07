"""Smoke tests for the local CGAN runtime and Flask API."""
import unittest
from io import BytesIO

import numpy as np
import torch

import app as shadowcomm


class CGANAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = shadowcomm.app.test_client()

    def test_home_page_renders_from_repository_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CGAN Demo", response.data)
        self.assertIn(b"no SDR measurements are connected", response.data)

    def test_models_load_and_metrics_are_not_fabricated(self):
        health = self.client.get("/api/config").get_json()["models"]
        self.assertTrue(health["generator_loaded"], health["load_errors"])
        self.assertTrue(health["decoder_loaded"], health["load_errors"])
        self.assertEqual(health["transport"], "offline_demo_only")
        self.assertTrue(health["manifest_matches_loaded_models"])

        report = self.client.get("/api/metrics").get_json()
        self.assertEqual(report["measurement_status"], "offline_report_loaded")
        self.assertIn("ks_pvalue", report)
        self.assertIn("adversary_accuracy", report)
        self.assertIn("ber", report)

    def test_long_message_uses_multiple_frames_and_round_trips(self):
        torch.manual_seed(12345)
        message = "CGAN integration loopback: " + ("payload " * 80)
        sent = self.client.post("/api/send", json={"message": message})
        self.assertEqual(sent.status_code, 200, sent.get_json())
        payload = sent.get_json()
        self.assertEqual(payload["status"], "generated")
        self.assertEqual(payload["transport"], "simulated_only_not_transmitted")
        self.assertGreater(payload["frame_count"], 8)
        self.assertEqual(payload["waveform_shape"], [payload["frame_count"], 2, 512])

        waveform_response = self.client.get(payload["waveform_url"])
        self.assertEqual(waveform_response.status_code, 200)
        waveform = np.load(BytesIO(waveform_response.data), allow_pickle=False)
        self.assertEqual(waveform.shape, tuple(payload["waveform_shape"]))
        self.assertTrue(np.isfinite(waveform).all())

        received = self.client.post("/api/receive", json={"message_id": payload["message_id"]})
        self.assertEqual(received.status_code, 200, received.get_json())
        self.assertEqual(received.get_json()["message"], message)
        self.assertEqual(received.get_json()["transport"], "simulated_model_loopback_not_radio")

    def test_aes_key_is_not_returned_by_config(self):
        config = self.client.get("/api/config").get_json()["config"]
        self.assertNotIn("aes_key", config)
        self.assertTrue(config["aes_key_set"])


if __name__ == "__main__":
    unittest.main()
