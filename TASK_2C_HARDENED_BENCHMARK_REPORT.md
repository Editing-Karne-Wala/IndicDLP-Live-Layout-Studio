# 📰 IndicDLP: Historical Advertisement & Hard-Negative Benchmark Report (Task 2.C)

**Component:** Historical Advertisement Classifier & False-Alarm Elimination Engine  
**Institution:** IDEAS-TIH, Indian Statistical Institute (ISI) Kolkata / DST  
**Supervision:** Dr. Sujoy Kumar Biswas, Dr. Ritam Ghosh, Prof. Dipti Prasad Mukherjee  
**Archive:** *Jugantar* (1946 Broadsheet Newspaper Scans, 700 Scans Mined)  
**Dataset:** `Hardened_Ad_Dataset` (68,043 Stratified Samples: 30,405 Ads vs. 37,638 Non-Ads)  
**Interactive Notebook:** [`hardened_ad_classifier_benchmark.ipynb`](file:///C:/Users/shiny/Documents/IndicDLP-Live-Layout-Studio/hardened_ad_classifier_benchmark.ipynb)  
**Best Production Checkpoint:** [`checkpoints/hardened_best_resnet18.pt`](file:///C:/Users/shiny/Documents/IndicDLP-Live-Layout-Studio/checkpoints/hardened_best_resnet18.pt)  
**Timestamp:** 2026-11-25 04:32:00 IST  
**GitHub Release:** [`release/pipeline-and-benchmark`](https://github.com/Editing-Karne-Wala/IndicDLP-Live-Layout-Studio/tree/release/pipeline-and-benchmark)

---

## 1. Executive Summary & Problem Formulation
In historical broadsheet newspaper archives, standard deep vision models suffer from severe false positives: **they mistake boxed editorial articles, war bulletins, and rectangular communiques for print advertisements** simply because both possess rectangular borders and boxed typography.

To eliminate this vulnerability:
1. We mined **over 28,000 deceptive "Hard-Negative" boxed news articles** and tall single-column stories across 700 broadsheets (`batch_2` through `batch_8`).
2. We compiled and stratified a balanced **68,043-crop dataset** (80% train / 10% val / 10% test with random seed 42).
3. We set and tuned hyperparameters on the validation set, and benchmarked **LeNet-5 (1998 Academic Baseline)** against **ResNet-18 (Deep Residual Model)** using Cosine Annealing learning rate schedules and FP16 Mixed Precision on an RTX 2050 GPU.

---

## 2. Hyperparameter Setting & Validation Set Tuning

Hyperparameters were set and iteratively tuned on the **6,803-sample validation split** (`val/`):

1. **Learning Rate & Optimization:**
   * **ResNet-18:** Initial learning rate $\eta = 1.5 \times 10^{-4}$ using the `AdamW` optimizer with weight decay $\lambda = 10^{-4}$. A **Cosine Annealing Learning Rate Scheduler** (`T_max = 4`) decayed the learning rate smoothly to zero over the 4 epochs.
   * **LeNet-5:** Initial learning rate $\eta = 3.0 \times 10^{-4}$ with `AdamW`.
2. **Loss Function & Class Weighting:**
   * Weighted Cross-Entropy Loss to handle subtle class distribution:
     $$w_{\text{ad}} = \frac{N_{\text{non\_ad}}}{N_{\text{ad}}} = \frac{30,110}{24,324} \approx 1.238, \quad w_{\text{non\_ad}} = 1.0$$
3. **Regularization & Batch Geometry:**
   * Batch size $B = 64$ with 4 asynchronous dataloader workers on CUDA:0 (RTX 2050).
   * Dropout ($p = 0.3$) applied to the penultimate classification head to prevent co-adaptation on scan artifacts.
4. **Validation-Guided Model Checkpointing:**
   * Models were evaluated at the end of each epoch on all 6,803 validation images. Checkpoints (`hardened_best_resnet18.pt` and `hardened_best_lenet.pt`) were saved only when the validation F1-score attained a new global maximum.
   * **ResNet-18 Validation Trajectory:** Epoch 1 ($F_1 = 0.8388$) $\to$ Epoch 2 ($F_1 = 0.8402$) $\to$ Epoch 3 ($F_1 = \mathbf{0.8505}$).
5. **Decision Threshold Calibration:**
   * Tuned on the validation ROC curve from standard $p = 0.50 \to p = 0.70$. This suppressed borderline false alarms while maintaining $>87\%$ true advertisement recall.

---

## 3. Evaluation on Held-Out Test Set (6,806 Unseen Test Crops)

Once optimal checkpoints were locked based on validation F1, final evaluation was performed on the **completely unseen, held-out Test Set (6,806 real historical crops)**:

| Evaluation Metric | LeNet-5 (Classical Baseline) | ResNet-18 (Deep Residual Model) | Performance Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 82.41% | **85.01%** | **+2.60% overall gain** |
| **Precision** | 75.47% | **80.48%** | **+5.01% precision boost** |
| **Recall (True Ads)** | **89.84%** | 87.73% | High capture rate of complex classified text |
| **F1-Score** | 0.8203 | **0.8395** | **+0.0192 balanced score** |
| **ROC-AUC** | 0.9130 | **0.9334** | **Superior class separation margin** |
| **False Positives (FP)** | 888 news articles | **647 news articles** | **▼ 27.1% Reduction in False Alarms!** |

### Confusion Matrix Breakdown:
* **LeNet-5 (1998):**
  * True Negatives (TN): **2,877**
  * False Positives (FP): **888** (Bordered news misclassified as ads)
  * False Negatives (FN): **309**
  * True Positives (TP): **2,732**
* **ResNet-18 (Residual Backbone):**
  * True Negatives (TN): **3,118** (+241 correct news identifications)
  * False Positives (FP): **647** (241 fewer news drops; **27.1% reduction**)
  * False Negatives (FN): **373**
  * True Positives (TP): **2,668**

---

## 4. Dataset Composition (`Hardened_Ad_Dataset`)

* **Total Samples:** 68,043
* **True Advertisements (`ad`):** 30,405
* **Non-Ads & Hard Negatives (`non_ad`):** 37,638
  * Boxed editorial announcements & bulletins: ~28,285
  * Extreme aspect ratio single-column news ($H/W > 4.5$): ~1,262
  * Standard multi-paragraph body text: ~8,091
* **Stratified Splits (Random Seed = 42):**
  * **Train (80%):** 54,434 crops (24,324 ads / 30,110 non-ads)
  * **Val (10%):** 6,803 crops (3,040 ads / 3,763 non-ads)
  * **Test (10%):** 6,806 crops (3,041 ads / 3,765 non-ads)
* **Manifest:** [`C:\Users\shiny\Desktop\Hardened_Ad_Dataset\dataset_manifest.json`](file:///C:/Users/shiny/Desktop/Hardened_Ad_Dataset/dataset_manifest.json)

---

## 5. Real Newspaper Broadsheet Scan Test
In Section 4 and 5 of [`hardened_ad_classifier_benchmark.ipynb`](file:///C:/Users/shiny/Documents/IndicDLP-Live-Layout-Studio/hardened_ad_classifier_benchmark.ipynb), we tested the production checkpoint on a full raw broadsheet page (`1946-01-14_page_005.jpg`):
* **YOLOv10x Proposals:** 66 structural regions detected.
* **ResNet-18 Disambiguation ($p \ge 0.70$):**
  * **38 True Ads** detected and cleanly framed in **Green**.
  * **28 Editorial News Columns & Boxed Announcements** preserved in **Blue** without accidental deletion or dropping.

---

## 6. Citations & Foundational Sources

The methodology, models, and layout architectures used in this evaluation benchmark are built upon the following academic and open-source foundations:

1. **Deep Residual Learning for Image Recognition (ResNet-18):**
   * *Citation:* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778.
   * *Role:* Provides the 18-layer residual skip-connection backbone enabling high receptive field feature extraction without gradient degradation.
   * *Source:* [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

2. **Gradient-Based Learning Applied to Document Recognition (LeNet-5 Baseline):**
   * *Citation:* LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), pp. 2278–2324.
   * *Role:* Serves as the classical historical academic CNN baseline to demonstrate why shallow networks fail to separate fine typography from framed news text.
   * *Source:* [http://vision.stanford.edu/cs598g_spring07/papers/Lecun98.pdf](http://vision.stanford.edu/cs598g_spring07/papers/Lecun98.pdf)

3. **IndicDLP: A Large-Scale Dataset for Indic Document Layout Parsing:**
   * *Citation:* AI4Bharat Research Consortium & IDEAS-TIH, ISI Kolkata (2025). IndicDLP: Benchmark and Fine-Tuned Detectors for 22 Scheduled Indian Languages. *arXiv preprint arXiv:2512.20236*.
   * *Role:* Core project schema, layout region taxonomy (42 classes), and fine-tuned detector weights (`doclayout_yolo_indicdlp.pt` and `md2_PT_indicdlp_FT.pt`).
   * *Project Page:* [https://indicdlp.github.io/](https://indicdlp.github.io/) | *Hugging Face:* [https://huggingface.co/datasets/ai4bharat/indicdlp](https://huggingface.co/datasets/ai4bharat/indicdlp)

4. **DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic and Natural Data:**
   * *Citation:* Zhao, Y., et al. (2024). DocLayout-YOLO: Enhancing Document Layout Analysis. *OpenDataLab Technical Report*.
   * *Role:* Global-to-local perception model used for initial structural region proposals.
   * *Source:* [https://github.com/opendatalab/DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

5. **Weighted Boxes Fusion (WBF): Ensembling Object Detection Models:**
   * *Citation:* Solovyev, R., Wang, W., & Gabruseva, T. (2021). Weighted boxes fusion: Ensembling methods for object detection. *Image and Vision Computing*, 107, 104117.
   * *Role:* Multi-scale inference and bounding box fusion across overlapping passes.
   * *Source:* [https://github.com/ZFTurbo/Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)

6. **Historical Archive Primary Source:**
   * *Archive:* *Jugantar* (যুগান্তর) Daily Broadsheet Newspaper, Year 1946 Collection, National Library of India & West Bengal Archives.
