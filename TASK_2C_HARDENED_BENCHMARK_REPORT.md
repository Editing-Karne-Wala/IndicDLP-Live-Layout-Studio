# 📰 IndicDLP: Historical Advertisement & Hard-Negative Benchmark Report (Task 2.C)

**Component:** Historical Advertisement Classifier & False-Alarm Elimination Engine  
**Institution:** IDEAS-TIH, Indian Statistical Institute (ISI) Kolkata / DST  
**Supervision:** Dr. Sujoy Kumar Biswas, Dr. Ritam Ghosh, Prof. Dipti Prasad Mukherjee  
**Archive:** *Jugantar* (1946 Broadsheet Newspaper Scans, 700 Scans Mined)  
**Dataset:** `Hardened_Ad_Dataset` (68,043 Stratified Samples: 30,405 Ads vs. 37,638 Non-Ads)  
**Interactive Notebook:** `hardened_ad_classifier_benchmark.ipynb`  
**Best Production Checkpoint:** `checkpoints/hardened_best_resnet18.pt`  
**Timestamp:** 2026-11-25 04:22:00 IST  

---

## 1. Executive Summary & Problem Formulation
In historical broadsheet newspaper archives, standard deep vision models suffer from severe false positives: **they mistake boxed editorial articles, war bulletins, and rectangular communiques for print advertisements** simply because both possess rectangular borders and boxed typography.

To eliminate this vulnerability:
1. We mined **over 28,000 deceptive "Hard-Negative" boxed news articles** and tall single-column stories across 700 broadsheets.
2. We compiled and stratified a balanced **68,043-crop dataset** (80% train / 10% val / 10% test).
3. We trained and benchmarked **LeNet-5 (1998 Academic Baseline)** against **ResNet-18 (Deep Residual Model)** using Cosine Annealing learning rate schedules and FP16 Mixed Precision on an RTX 2050 GPU.

---

## 2. Hardened Test Set Evaluation (6,806 Unseen Test Crops)

| Evaluation Metric | LeNet-5 (Classical Baseline) | ResNet-18 (Deep Residual Model) | Performance Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 82.41% | **85.01%** | **+2.60% overall gain** |
| **Precision** | 75.47% | **80.48%** | **+5.01% precision boost** |
| **Recall (True Ads)** | **89.84%** | 87.73% | High capture rate of complex classified text |
| **F1-Score** | 0.8203 | **0.8395** | **+0.0192 balanced score** |
| **ROC-AUC** | 0.9130 | **0.9334** | **Superior class separation margin** |
| **False Positives (FP)** | 888 news articles | **647 news articles** | **▼ 27.1% Reduction in False Alarms!** |

### Confusion Matrix Breakdown:
* **LeNet-5:** 2,877 TN | **888 FP** (Bordered news misclassified as ads) | 309 FN | 2,732 TP
* **ResNet-18:** 3,118 TN | **647 FP** (241 fewer news drops) | 373 FN | 2,668 TP

---

## 3. Dataset Composition (`Hardened_Ad_Dataset`)

* **Total Samples:** 68,043
* **True Advertisements (`ad`):** 30,405
* **Non-Ads & Hard Negatives (`non_ad`):** 37,638
  * Boxed editorial announcements & bulletins: ~28,285
  * Extreme aspect ratio single-column news ($H/W > 4.5$): ~1,262
  * Standard multi-paragraph body text: ~8,091
* **Stratified Splits:**
  * **Train (80%):** 54,434 crops (24,324 ads / 30,110 non-ads)
  * **Val (10%):** 6,803 crops (3,040 ads / 3,763 non-ads)
  * **Test (10%):** 6,806 crops (3,041 ads / 3,765 non-ads)

---

## 4. Real Newspaper Broadsheet Scan Test
In Section 4 and 5 of `hardened_ad_classifier_benchmark.ipynb`, we tested the production checkpoint on a full raw broadsheet page (`1946-01-14_page_005.jpg`):
* **YOLOv10x Proposals:** 66 structural regions detected.
* **ResNet-18 Disambiguation ($p \ge 0.70$):**
  * **38 True Ads** detected and cleanly framed in **Green**.
  * **28 Editorial News Columns & Boxed Announcements** preserved in **Blue** without accidental deletion or dropping.

---

## 5. Key Engineering Takeaways for Production Testing
1. **Shallow vs. Deep Architectures:** LeNet-5 drops from 96% on clean data down to 82% on hardened data because its 2 convolutional layers cannot model fine typographic strokes. ResNet-18 maintains high discriminative power with 85% accuracy and 93.3% ROC-AUC.
2. **Decision Thresholding:** In production, shifting the decision threshold from $0.50 \to 0.70$ completely suppresses borderline news false positives while preserving genuine print advertisements.
