# 專案歷程與實驗結果紀錄

> 最後更新：2026-05-18

---

## 一、資料集概覽

**來源**：ADNI（Alzheimer's Disease Neuroimaging Initiative）T1 結構式 MRI

### 原始資料（Exp 1–4，少量資料）

| 類別 | 唯一受試者數 | 掃描總數 |
|------|------------|--------|
| AD（阿茲海默症） | 31 | 64 |
| NC（正常對照） | 43 | 87 |
| MCI（輕度認知障礙） | 63 | 131 |
| **合計** | **137** | **282** |

### 擴充資料後（Exp 5，2026-05-17）

| 類別 | 唯一受試者數 | 掃描總數 |
|------|------------|--------|
| AD | 88 | 290 |
| NC | 385 | 511 |
| MCI | 241 | 512 |
| **合計** | **714** | **1,313** |

**Task 1（AD vs NC）**：801 scans / 473 subjects（Exp 6–9 使用）  
**Task 2（MCI 轉換 sMCI vs pMCI）**：131 scans / 63 subjects（MCI 資料**未擴充**）

**前處理流程**（`preprocess.py`）
1. FreeSurfer recon-all skull-stripping
2. 重採樣至 1mm isotropic
3. 大腦 CoM 對齊，resize 至 128×128×128 voxels
4. z-score 標準化（以受試者自身腦組織為基準）
5. 存為 `.npy`，背景值設為 0.0

**已知資料問題**
- 大腦在 128³ 體積中並非完全置中（組織邊界約 [20,19,21]→[127,127,127]）
- z-score 後頭骨剝離邊界產生強烈負值條紋（≈ −1.9），在 Grad-CAM 中顯示為藍色偽影
- AD/NC 比例失衡（NC:AD ≈ 3:1）→ 模型偏向預測 NC，sensitivity@0.5 常為 0

---

## 二、模型演進與實驗結果

### 實驗框架
- 驗證：5-fold StratifiedGroupKFold（以 subject_id 分組，防止同一受試者跨折）
- 主要評估指標：**AUC-ROC**（其餘指標受 threshold calibration 影響）
- 輔助指標：Accuracy, Sensitivity, Specificity, **Youden's J optimal threshold**（已加入 compute_metrics）
- 硬體：NVIDIA RTX 3050 4GB + CUDA 13.0
- Python 環境：`.venv/`（torch 2.10.0+cu130, MONAI）

**指標說明**
- Sensitivity@0.5 常為 0：NC:AD=3:1 不平衡導致模型機率分布偏低，並非訓練失敗
- Youden's J = max(Sensitivity + Specificity − 1)：找最佳分類門檻，用於不平衡資料集

---

### Exp 1 — Baseline 3D CNN（`train_baseline.py`，小資料）

**模型**：`models/baseline_cnn.py`（~200K 參數）  
3D Conv×3 + MaxPool + GlobalAvgPool + Linear

**訓練設定**：CE loss，Adam lr=1e-3，epochs=50，batch=4

**結果（5-fold CV，74 subjects）**

| 指標 | Mean | Std |
|------|------|-----|
| Accuracy | 0.534 | 0.139 |
| Sensitivity | 0.405 | 0.319 |
| Specificity | 0.616 | 0.248 |
| **AUC** | **0.564** | **0.204** |

**結論**：接近隨機（AUC≈0.5），模型過於簡單。

---

### Exp 2 — DenseNet121（`archive/train_densenet.py`，小資料）

**模型**：`models/densenet_monai.py`（MONAI 內建 DenseNet121，~11.2M 參數）

**訓練設定**：CE + class weight，AdamW lr=1e-4，ReduceLROnPlateau，epochs=100，batch=2

**結果（5-fold CV，74 subjects）**

| 指標 | Mean | Std |
|------|------|-----|
| Accuracy | 0.590 | 0.086 |
| Sensitivity | 0.402 | 0.484 |
| Specificity | 0.711 | 0.323 |
| **AUC** | **0.580** | **0.102** |

**結論**：11.2M params vs 74 subjects → 嚴重過擬合，折間方差極高。

---

### Exp 3 — LightCNN3D v1（Focal Loss + OneCycleLR，小資料）

**模型**：`models/light_cnn3d.py`（新設計，~587K 參數）
- Stem: Conv(1→16, 7³, stride=2) + BN + ReLU
- Stage1–3: ResBlock + DownConv（16→32→64→128）
- Head: GlobalAvgPool + Dropout(0.5) + Linear(128→2)

**訓練設定**：Focal loss（γ=2, AD weight=2.5, label_smoothing=0.1），AdamW lr=3e-4，OneCycleLR，epochs=80

**結果（5-fold CV）**

| 指標 | Mean | Std |
|------|------|-----|
| Accuracy | 0.472 | 0.106 |
| Sensitivity | 0.938 | 0.085 |
| Specificity | 0.128 | 0.229 |
| **AUC** | **0.618** | **0.179** |

**結論**：Focal + class weight 過積極 → 崩潰成全預測 AD。

---

### Exp 4 — LightCNN3D v2（CE Loss + CosineAnnealing，小資料）

同 LightCNN3D 架構，改用 CE loss，無 class weight，CosineAnnealingLR

**結果（5-fold CV）**

| Fold | Best Epoch | AUC |
|------|-----------|-----|
| 1 | 18 | 0.858 |
| 2 | 2 | 0.745 |
| 3 | 6 | 0.532 |
| 4 | 35 | 0.393 |
| 5 | 39 | 0.540 |

| 指標 | Mean | Std |
|------|------|-----|
| **AUC** | **0.614** | **0.186** |

**結論**：資料量過少（74 subjects），折 4、5 崩潰。需更多資料。

---

### Exp 5 — 資料擴充（2026-05-17）

從 ADNI 下載更多掃描，執行 `preprocess.py` 重新前處理：
- AD+NC 從 151 scans / 74 subjects → **801 scans / 473 subjects**
- MCI 資料維持 131 scans / 63 subjects（MCI 轉換預測資料未擴充）
- `augment_offline.py --n_aug 5 --groups AD NC` → augmented_list.csv（備用，實際訓練未使用）

---

### Exp 6 — Baseline 3D CNN v2（擴充資料）

同 Exp 1 架構（~200K 參數），801 AD+NC scans。

**結果（5-fold CV，`results/cv_baseline_v2_results.csv`）**

| Fold | Best Epoch | Accuracy | Sensitivity | Specificity | AUC |
|------|-----------|----------|-------------|-------------|-----|
| 1 | 32 | 0.745 | 0.585 | 0.800 | 0.792 |
| 2 | 40 | 0.825 | 0.700 | 0.867 | **0.847** |
| 3 | 17 | 0.700 | 0.400 | 0.800 | 0.576 |
| 4 | 1 | 0.750 | 0.000 | 1.000 | 0.638 |
| 5 | 7 | 0.744 | 0.000 | 1.000 | 0.639 |

| 指標 | Mean | Std |
|------|------|-----|
| Accuracy | 0.753 | 0.045 |
| Sensitivity | 0.337 | 0.326 |
| Specificity | 0.893 | 0.101 |
| **AUC** | **0.698** | **0.115** |

**結論**：多資料讓 AUC 從 0.564 提升至 0.698，但 fold 4、5 sensitivity=0。

---

### Exp 7 — LightCNN3D v3（擴充資料，CE loss）

CE loss，CosineAnnealingLR，dropout=0.5，AdamW lr=1e-4，epochs=100，patience=25

**結果（5-fold CV，`results/cv_light_v3_results.csv`）**

| Fold | Best Epoch | Accuracy | Sensitivity | Specificity | AUC |
|------|-----------|----------|-------------|-------------|-----|
| 1 | 8 | 0.739 | 0.585 | 0.792 | 0.734 |
| 2 | 4 | 0.806 | 0.700 | 0.842 | **0.836** |
| 3 | 4 | 0.644 | 0.450 | 0.708 | 0.634 |
| 4 | 11 | 0.725 | 0.450 | 0.817 | 0.711 |
| 5 | 5 | 0.675 | 0.488 | 0.739 | 0.691 |

| 指標 | Mean | Std |
|------|------|-----|
| Accuracy | 0.718 | 0.063 |
| Sensitivity | 0.535 | 0.108 |
| Specificity | 0.780 | 0.055 |
| **AUC** | **0.721** | **0.074** |

---

### Exp 8 — MCI 轉換預測（DenseNet121，已被 Exp 10 取代）

DenseNet121 (11.2M params) 對 63 subjects MCI 資料；AUC ≈ 0.571，AUC 接近隨機。
模型過大導致過擬合。已由 Exp 10 LightCNN3D 版本取代。

---

### Exp 9 — LightCNN3D v4（⭐ 當前最佳 AD/NC）

**核心改動（vs v3）**
1. **Loss**：CE loss → **Focal Loss only**（γ=2.0，`ad_weight=1.0` 移除 class weight）
2. **Checkpointing**：以 val_loss 為基準 → **以 val_AUC 為基準**（`if auc > best_val_auc + min_delta`）
3. **Youden's J**：`compute_metrics` 加入最佳 threshold（youden_threshold, youden_sensitivity, youden_specificity）

**訓練設定**：AdamW lr=1e-4, weight_decay=1e-4，CosineAnnealingLR（T_max=150，eta_min=1e-6），dropout=0.2，epochs=150，patience=35，batch=4

**結果（5-fold CV，`results/cv_light_v4_results.csv`）**

| Fold | Best Epoch | Sensitivity@0.5 | Specificity@0.5 | AUC |
|------|-----------|-----------------|-----------------|-----|
| 1 | 41 | 0.000 | 1.000 | 0.770 |
| 2 | 6 | 0.000 | 1.000 | **0.839** |
| 3 | 47 | 0.400 | 0.783 | 0.727 |
| 4 | 17 | 0.400 | 0.833 | 0.714 |
| 5 | 33 | 0.390 | 0.798 | 0.704 |

| 指標 | Mean | Std |
|------|------|-----|
| Accuracy | 0.720 | 0.029 |
| Sensitivity@0.5 | 0.238 | 0.217 |
| Specificity@0.5 | 0.883 | 0.108 |
| **AUC** | **0.751** | **0.056** |

> Fold 1、2 sensitivity@0.5=0：門檻校正問題，非訓練失敗。Youden's J 可找回最佳門檻。

**最佳模型**：`checkpoints/best_light.pth`（= fold 2, epoch 6, AUC=0.839）

**與前版本比較**

| 指標 | Baseline v2 | LightCNN3D v3 | LightCNN3D v4 |
|------|------------|--------------|--------------|
| AUC | 0.698±0.115 | 0.721±0.074 | **0.751±0.056** |
| Sensitivity | 0.337±0.326 | 0.535±0.108 | 0.238±0.217 |
| Specificity | 0.893±0.101 | 0.780±0.055 | 0.883±0.108 |
| AUC std ↓ | — | ▼ vs baseline | ▼ 最低方差 |

> v4 在 sensitivity@0.5 反而較低，因 focal loss 造成機率更偏向 NC；但 AUC（門檻無關）提升。

---

### Exp 10 — LightCNN3D MCI v2（⭐ 當前最佳 MCI 轉換）

**核心改動（vs Exp 8 DenseNet121）**
1. **模型**：DenseNet121 (11.2M params) → **LightCNN3D (587K params)**（適合小資料集）
2. **Scheduler**：ReduceLROnPlateau → **CosineAnnealingLR**（T_max=150）
3. **Checkpointing**：val_loss → **val_AUC**
4. **Class weight**：`pmci_weight=1.5`（因 pMCI:sMCI ≈ 1:1 輕微失衡）

**訓練設定**：AdamW lr=1e-4, weight_decay=1e-4，dropout=0.2，epochs=150，patience=35，batch=2，131 scans / 63 subjects

**結果（5-fold CV，`results/cv_mci_v2_results.csv`）**

| Fold | Best Epoch | Sensitivity@0.5 | Specificity@0.5 | AUC |
|------|-----------|-----------------|-----------------|-----|
| 1 | 1 | 0.000 | 1.000 | 0.761 |
| 2 | 40 | 0.857 | 0.308 | 0.709 |
| 3 | 26 | 1.000 | 0.182 | **0.867** |
| 4 | 58 | 1.000 | 0.000 | 0.740 |
| 5 | 3 | 1.000 | 0.000 | 0.643 |

| 指標 | Mean | Std |
|------|------|-----|
| Accuracy | 0.557 | 0.080 |
| Sensitivity@0.5 | 0.771 | 0.408 |
| Specificity@0.5 | 0.298 | 0.404 |
| **AUC** | **0.744** | **0.082** |

> Fold 4、5 specificity@0.5=0（模型偏向全預測 pMCI）；AUC 正確反映分類能力。  
> AUC 0.744 對 63 subjects 屬合理表現，文獻 SOTA ≈ 0.75–0.80（使用數百至數千受試者）。

**最佳模型**：`checkpoints/best_mci.pth`（= fold 3, epoch 26, AUC=0.867）

**與 Exp 8 DenseNet121 比較**

| 指標 | DenseNet121 (Exp 8) | LightCNN3D v2 (Exp 10) |
|------|---------------------|------------------------|
| AUC | 0.571±0.141 | **0.744±0.082** |
| Sensitivity | 0.987±0.030 | 0.771±0.408 |
| Specificity | 0.103±0.151 | 0.298±0.404 |

---

## 三、Grad-CAM 可視化

**工具**：`visualize_gradcam.py`（MONAI GradCAMpp；fallback 為 input-gradient saliency）

### Target Layer 選擇

| 模型 | Layer | 空間解析度 | 說明 |
|------|-------|-----------|------|
| DenseNet121 | `features.denseblock4` | — | MONAI 原生 |
| LightCNN3D (old) | `stage3.0` | 8³ | 解析度過低 |
| **LightCNN3D (current)** | **`stage2.0`** | **32³** | ⭐ 更細緻空間資訊 |

### 解剖位置估算（`--anatomical` 旗標）

無 MNI 配准，改用**受試者自身腦邊界框**（非零 voxels 的 z 軸範圍）按比例取切面：

| 解剖位置 | z 比例 | AD 臨床意義 |
|---------|--------|-----------|
| Entorhinal Cortex | 25% | 最早出現萎縮 |
| Hippocampus / Amygdala | 38% | **MCI→AD 關鍵預測指標** |
| Mid-Temporal | 52% | 顳葉本體 |
| Temporoparietal | 68% | 晚期 AD 擴散區 |

估算精度 ±8–12 voxels（無 MNI 配准的固有誤差），足以確認模型是否關注正確腦區。

### AD/NC — LightCNN3D v4（`results/gradcam_light_v4/`，6 subjects）

基於 `best_light.pth`（fold 2, epoch 6, AUC=0.839）：

| 受試者 | 真實 | 預測 | 信心度 | 解剖觀察 |
|--------|------|------|--------|---------|
| 005_S_0553 | NC | NC | — | 廣泛激活，符合全腦保留 |
| 005_S_0602 | NC | NC | — | 廣泛激活，符合全腦保留 |
| 007_S_1206 | NC | NC | — | 廣泛激活 |
| 005_S_0814 | AD | NC | — | 誤分類；熱圖廣泛分布 |
| 007_S_1304 | AD | NC | — | 誤分類；熱圖廣泛分布 |
| 013_S_0996 | AD | AD | — | 顳葉前下方有局部亮點 ✓ |

### AD/NC — Baseline v2（`results/gradcam_baseline_v4/`，6 subjects）

對比實驗，使用 `best_baseline.pth`。

### MCI 轉換 — LightCNN3D MCI v2（`results/gradcam_mci_v3/`，6 subjects + 1 比較圖）

基於 `best_mci.pth`（fold 3, epoch 26, AUC=0.867）：

| 受試者 | 標籤 | 檔案 |
|--------|------|------|
| 005_S_0324 | sMCI | `005_S_0324_sMCI_anatomical.png` |
| 005_S_0448 | pMCI | `005_S_0448_pMCI_anatomical.png` |
| 005_S_0572 | pMCI | `005_S_0572_pMCI_anatomical.png` |
| 016_S_0769 | pMCI | `016_S_0769_pMCI_anatomical.png` |
| 016_S_1149 | sMCI | `016_S_1149_sMCI_anatomical.png` |
| 023_S_0613 | sMCI | `023_S_0613_sMCI_anatomical.png` |

**pMCI vs sMCI 並排比較圖**：`results/gradcam_mci_v3/comparison_pmci_vs_smci.png`  
列：pMCI subjects（左）｜ sMCI subjects（右）  
行：Entorhinal → Hippocampus → Mid-Temporal → Temporoparietal

### 已知偽影
- 頭骨剝離邊界（z-score 後 ≈ −1.9 強負值條紋）在 Grad-CAM 顯示為藍色垂直條
- 修復方向：preprocess.py 邊界處理，或在 dataset.py 截斷極端負值

---

## 四、最終結果總覽

### Task 1：AD vs NC（5-fold CV，801 scans / 473 subjects）

| 模型 | AUC | Std | 備注 |
|------|-----|-----|------|
| Baseline v2 | 0.698 | 0.115 | 高方差，2 折崩潰 |
| LightCNN3D v3 | 0.721 | 0.074 | 穩定，無崩潰 |
| **LightCNN3D v4** | **0.751** | **0.056** | ⭐ 最佳，最低方差 |

### Task 2：MCI 轉換 sMCI vs pMCI（5-fold CV，131 scans / 63 subjects）

| 模型 | AUC | Std | 備注 |
|------|-----|-----|------|
| DenseNet121 (Exp 8) | 0.571 | 0.141 | 幾乎隨機，過擬合 |
| **LightCNN3D v2** | **0.744** | **0.082** | ⭐ 最佳，+0.173 |

---

## 五、檔案樹（2026-05-18 整理版）

```
final projecet/
├── docs/
│   ├── history_and_results.md        ← 本文件
│   └── next_steps.md
│
├── models/
│   ├── baseline_cnn.py               # Exp 1（保留供參考）
│   ├── densenet_monai.py             # Exp 2（保留供參考）
│   ├── light_cnn3d.py               # ⭐ Exp 3–10 使用（~587K params）
│   └── __init__.py
│
├── data/
│   ├── processed/
│   │   ├── AD/                       # 290 scans .npy
│   │   ├── NC/                       # 511 scans .npy
│   │   └── MCI/                      # 512 scans .npy
│   ├── processed_list.csv            # AD+NC+MCI 掃描清單
│   ├── mci_conversion_list.csv       # MCI 轉換任務（131 rows，63 subjects）
│   ├── subject_list.csv
│   └── quality_report.csv
│
├── dataset/ADNI/                     ← 原始 ADNI 下載檔（MP-RAGE T1）
│
├── checkpoints/
│   ├── best_light.pth               # ⭐ LightCNN3D v4 最佳（fold2 ep6 AUC=0.839）
│   ├── best_light_fold[1-5].pth
│   ├── best_mci.pth                 # ⭐ LightCNN3D MCI v2 最佳（fold3 ep26 AUC=0.867）
│   ├── best_mci_fold[1-5].pth
│   ├── best_baseline.pth            # Baseline v2 最佳（fold2 AUC=0.847）
│   └── best_baseline_fold[1-5].pth
│
├── results/
│   ├── cv_light_v4_results.csv          # ⭐ LightCNN3D v4 各折詳細
│   ├── cv_light_v4_results_summary.csv  # ⭐ LightCNN3D v4 摘要
│   ├── cv_mci_v2_results.csv            # ⭐ LightCNN3D MCI v2 各折詳細
│   ├── cv_mci_v2_results_summary.csv    # ⭐ LightCNN3D MCI v2 摘要
│   ├── cv_light_v3_results*.csv         # LightCNN3D v3（保留）
│   ├── cv_baseline_v2_results*.csv      # Baseline v2（保留）
│   ├── training_log_light_fold[1-5].csv
│   ├── training_log_mci_fold[1-5].csv
│   ├── training_log_baseline.csv
│   ├── norm_stats_light_fold[1-5].json
│   ├── norm_stats_mci_fold[1-5].json
│   ├── figures_light_v4/               # ⭐ v4 報告圖表（learning curves, AUC, CM）
│   ├── figures_mci_v2/                 # ⭐ MCI v2 報告圖表
│   ├── figures_light/                  # v3 圖表（保留）
│   ├── figures_baseline/               # Baseline 圖表（保留）
│   ├── gradcam_light_v4/              # ⭐ LightCNN3D v4 Grad-CAM（6+6 PNGs）
│   ├── gradcam_mci_v3/                # ⭐ MCI v2 Grad-CAM（6+6+1 comparison PNG）
│   ├── gradcam_baseline_v4/           # Baseline Grad-CAM（對比用）
│   ├── gradcam_baseline_v3/           # 舊版 Baseline Grad-CAM
│   ├── gradcam_light_v2/              # 舊版 LightCNN3D Grad-CAM
│   └── gradcam_mci_v2/                # 舊版 MCI Grad-CAM
│
├── scripts/archive/                  ← 一次性腳本（已完成任務，不再使用）
│   ├── train_densenet.py             # Exp 2（DenseNet，已由 LightCNN3D 取代）
│   ├── train_mci_multichannel.py     # MCI 3-channel 實驗（已淘汰）
│   ├── evaluate.py                   # 舊版獨立評估腳本
│   ├── rebuild_csv.py                # 重建 processed_list.csv（已完成）
│   ├── merge_labels.py               # ADNI metadata 標籤合併（已完成）
│   ├── prepare_mci_labels.py         # MCI 轉換標籤（已完成）
│   └── prepare_metadata.py           # metadata 整理（已完成）
│
├── dataset.py                        # 資料載入 / ADNINpyDataset
├── preprocess.py                     # 原始 NIFTI → .npy 前處理
├── augment_offline.py                # 離線資料擴增（備用）
├── train_baseline.py                 # Exp 6 訓練腳本
├── train_light.py                   # ⭐ Exp 7/9 AD/NC 主訓練腳本
├── train_mci_conversion.py           # ⭐ Exp 10 MCI 轉換訓練腳本
├── visualize_gradcam.py             # ⭐ Grad-CAM（--anatomical / --compare）
├── plot_results.py                   # 報告圖表生成
├── adni_metadata.csv
├── requirements.txt                  # CPU 版
├── requirements-gpu-cu130.txt        # ⭐ GPU 版（實際使用）
├── setup_venv.ps1                    # 虛擬環境建立腳本
└── .venv/                            # Python 虛擬環境（torch 2.10.0+cu130）
```

---

## 六、已刪除的舊檔案紀錄

### 2026-05-14 整理（第一次）

#### Smoke test / 除錯輸出
- `checkpoints/smoke*`、`smoketest*/`、`tmp_smoke/`、`mci_smoke/`
- `results/smoketest*/`、`results/smoke_eval/`、`smoke_training_log*.csv`
- `results/cv_densenet_smoketest*.csv`、`cv_light_smoketest*.csv`
- `results/debug_*.png`、`results/brain_slices_debug.png`

#### 舊版 Grad-CAM（DenseNet，已被取代）
- `results/gradcam_adnc/`、`gradcam_adnc_final/`、`gradcam_adnc_fixed/`
- `results/gradcam_adnc_fixed2/`、`gradcam_adnc_new/`、`gradcam_adnc_v3/`

#### 舊版訓練 log
- `results/train_densenet_128.log`、`train_light*.log`、`auto_posttraining.log`

#### Baseline checkpoints（模型已被取代）
- `checkpoints/best_baseline.pth`（舊小資料版）
- `checkpoints/best_baseline_fold[1-5].pth`（舊小資料版）

### 2026-05-18 整理（第二次，~786 MB）

已刪除的大型 checkpoint 檔案（DenseNet 已完全由 LightCNN3D 取代）：
- `checkpoints/best_densenet.pth`（~44 MB）
- `checkpoints/best_densenet_fold[1-5].pth`（~220 MB 合計）
- `checkpoints/best_mci3ch.pth` + `best_mci3ch_fold[1-5].pth`（MCI 3-channel，已淘汰）
- 對應的 `results/cv_mci3ch_*.csv`、`training_log_mci3ch*.csv`、`norm_stats_mci3ch*.json`
- 舊版 `checkpoints/best_mci_fold[1-5].pth`（DenseNet MCI 版本）

**釋放空間**：約 786 MB
