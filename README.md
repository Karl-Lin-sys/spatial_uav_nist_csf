# Spatial 3D GIS & UAV Autonomous Navigation Core (NIST CSF 2.0 & Taiwan PDPA Compliant)

## 1. Overview
This repository implements the core algorithms for **3D GIS Level-of-Detail (LOD) Point Cloud Processing**, **Stereo Metric 3D Reconstruction**, **UAV Moving-Platform Landing Pose Estimation**, and **Balanced Softmax Long-Tail Loss**, built in strict accordance with:
- **NIST Cybersecurity Framework (CSF) 2.0** (`GOVERN`, `IDENTIFY`, `PROTECT`, `DETECT`, `RESPOND`, `RECOVER`)
- **Taiwan Personal Data Protection Act (PDPA)** & **Copyright Act** (zero Personal Identifiable Information or proprietary recruitment text is stored in this repository).

---

## 2. NIST CSF 2.0 Implementation Architecture

| NIST CSF 2.0 Function | Technical Implementation in This Repository |
| :--- | :--- |
| **GOVERN (GV)** | Enforces strict data minimization and open-source licensing. All recruitment chat logs, phone numbers, LINE IDs, and PDF resumes are prohibited from version control via `.gitignore` and automated pre-commit sanitization. |
| **IDENTIFY (ID)** | `TaiwanPDPAComplianceGuard` classifies Taiwanese phone numbers (`09xx-xxx-xxx`), landlines, emails, LINE IDs, and resume filenames as restricted PII under Taiwan's PDPA. |
| **PROTECT (PR)** | Uses `@torch.inference_mode()` and contiguous NumPy memory buffers (`np.ascontiguousarray`) to prevent GPU/CPU memory leaks and resource exhaustion (DoS). |
| **DETECT (DE)** | `TaiwanPDPAComplianceGuard.audit_and_redact()` scans text in $\mathcal{O}(L)$ linear time and automatically masks any accidentally injected PII before logging or export. |
| **RESPOND (RS) & RECOVER (RC)** | Automated unit tests (`test_spatial_uav_core.py`) verify zero PII leakage and mathematical correctness on every commit. |

---

## 3. Toddler-Friendly Explanation (Canadian English) 🇨🇦

> "Hey there, little buddy! Grab your toque and mittens, eh!
>
> Imagine we're building a giant 3D Lego town on the computer (**3D GIS**) and flying a smart toy helicopter (**UAV**) that has two robot eyes (**Stereo Vision**) so it can see how far away the trees and snowbanks are!
>
> 1. **Sorting the Lego Bricks (`SpatialPointCloudLOD`)**: When there are way too many tiny Lego dots on the screen, our computer squishes neighbours together into neat little blocks so the game runs super fast without getting tired!
> 2. **Robot Eyes & Safe Landing (`compute_metric_point_cloud_from_stereo` & `estimate_uav_landing_pose`)**: The helicopter looks with both eyes at the same time to measure the ground and land safely right on top of a moving toboggan!
> 3. **Playing Fair (`BalancedSoftmaxLoss`)**: Because the sky is super big and the landing pad is super tiny, we teach our robot brain to pay extra-special attention to the tiny landing pad so it never misses it!
> 4. **Keeping Secrets Safe (`TaiwanPDPAComplianceGuard` / NIST CSF 2.0)**: Before we share our toy plans on the internet (**GitHub**), our security helper checks that nobody's real name, phone number, or private mail gets shared—because in Taiwan, the law says we always protect our friends' privacy. Beauty, eh!"

---

## 4. Legal Upload to GitHub (Taiwanese Law Compliance)

Because all personal data (names, phone numbers, LINE IDs, resume filenames, and verbatim job postings) has been completely excluded and protected by `TaiwanPDPAComplianceGuard`, this repository is **100% legal to publish under Taiwan's Personal Data Protection Act (個人資料保護法) and Copyright Act (著作權法)**.
