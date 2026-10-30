"""
Verification & Compliance Test Suite for NIST CSF 2.0 + 3D GIS / UAV Pipeline.
"""

from __future__ import annotations

import unittest
import cv2
import numpy as np
import torch

from spatial_uav_core import (
    BalancedSoftmaxLoss,
    SpatialPointCloudLOD,
    TaiwanPDPAComplianceGuard,
    compute_metric_point_cloud_from_stereo,
    estimate_uav_landing_pose,
)


class TestSpatialUAVAndNISTCompliance(unittest.TestCase):
    def test_taiwan_pdpa_pii_redaction(self) -> None:
        sample_log = (
            "聯絡電話：0912345678, 電洽 07-1234567, "
            "我的line ID是synthetic_id_001, mail至 hr@example.com.tw, "
            "Applicant_Sample_Resume (1).pdf"
        )
        report = TaiwanPDPAComplianceGuard.audit_and_redact(sample_log)
        self.assertFalse(report.is_compliant)
        self.assertIn("TW_MOBILE_PHONE", report.detected_pii_types)
        self.assertIn("TW_LANDLINE_PHONE", report.detected_pii_types)
        self.assertIn("EMAIL_ADDRESS", report.detected_pii_types)
        self.assertIn("LINE_ID_MENTION", report.detected_pii_types)
        self.assertIn("RESUME_ATTACHMENT", report.detected_pii_types)
        self.assertNotIn("0912345678", report.redacted_text)
        self.assertNotIn("synthetic_id_001", report.redacted_text)
        self.assertNotIn("hr@example.com.tw", report.redacted_text)

    def test_spatial_point_cloud_lod(self) -> None:
        lod = SpatialPointCloudLOD(voxel_size=1.0)
        pts = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.3, 0.4, 0.5],  # Same voxel (0, 0, 0) -> centroid (0.2, 0.3, 0.4)
                [2.5, 2.5, 2.5],  # Separate voxel (2, 2, 2)
            ],
            dtype=torch.float32,
        )
        downsampled = lod(pts)
        self.assertEqual(downsampled.shape, (2, 3))
        self.assertFalse(downsampled.requires_grad)
        torch.testing.assert_close(
            downsampled[0], torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32)
        )

    def test_stereo_and_pnp_pipeline(self) -> None:
        # Synthetic textured stereo pair
        rng = np.random.default_rng(42)
        left = rng.integers(0, 255, size=(120, 160), dtype=np.uint8)
        right = np.roll(left, -8, axis=1)  # 8-pixel horizontal disparity

        cloud = compute_metric_point_cloud_from_stereo(
            left_gray=left,
            right_gray=right,
            focal_length_px=300.0,
            baseline_m=0.12,
            cx=80.0,
            cy=60.0,
        )
        self.assertEqual(cloud.ndim, 2)
        self.assertEqual(cloud.shape[1], 3)
        self.assertGreater(cloud.shape[0], 0)

        # Synthetic 3D landing pad corners & projected 2D coordinates
        obj_pts = np.array(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.25, -0.25, 0.0],
            ],
            dtype=np.float64,
        )
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        dist = np.zeros(5, dtype=np.float64)
        rvec_gt = np.array([[0.05], [-0.03], [0.1]], dtype=np.float64)
        tvec_gt = np.array([[0.1], [-0.2], [5.0]], dtype=np.float64)
        img_pts, _ = cv2.projectPoints(obj_pts, rvec_gt, tvec_gt, K, dist)

        ok, rvec_est, tvec_est = estimate_uav_landing_pose(
            obj_pts, img_pts.reshape(-1, 2), K, dist
        )
        self.assertTrue(ok)
        np.testing.assert_allclose(tvec_est, tvec_gt, atol=1e-3)

    def test_balanced_softmax_loss(self) -> None:
        counts = torch.tensor([10000.0, 50.0])  # Extreme 200:1 class imbalance
        criterion = BalancedSoftmaxLoss(counts, tau=1.0)
        logits = torch.tensor([[1.0, 1.0], [0.5, 2.0]], requires_grad=True)
        targets = torch.tensor([0, 1], dtype=torch.long)
        loss = criterion(logits, targets)
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
