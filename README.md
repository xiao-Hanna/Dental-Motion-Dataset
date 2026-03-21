# Dental-Motion-Dataset
A publicly available dental dataset containing 27 mandibular movement videos and over 1,200 manually annotated oral images, along with benchmark evaluations using mainstream segmentation and keypoint detection methods.
# Dental Keypoint Detection Dataset

## 📌 Overview

This dataset is designed for dental keypoint detection tasks. It provides annotated landmarks for dental images, supporting research in medical image analysis, computer vision, and mandibular motion tracking.

---

## 📊 Dataset Information

* **Total images:** 1200
* **Keypoints per image:** 35
* **Annotation formats:** COCO-style JSON and TXT
* **Train/Validation split:** 80% / 20%

---

## 📂 Dataset Structure

```
dataset/
├── images/
├── keypoints/
├── masks/
├── videos/
└── metadata.csv
```

* `images/`: original dental images
* `keypoints/`: landmark annotations (JSON / TXT)
* `masks/`: segmentation masks (if applicable)
* `videos/`: original video sequences (optional)
* `metadata.csv`: dataset index and train/val split

---

## 📥 Download

The full dataset is available on Zenodo:

👉 https://doi.org/10.5281/zenodo.19146909

---

## 🧠 Applications

This dataset can be used for:

* Dental keypoint detection
* Landmark localization
* Mandibular motion analysis
* Medical image understanding
* Deep learning model training

---

## 📄 License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License.

---
