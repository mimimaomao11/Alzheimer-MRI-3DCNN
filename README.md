# ADNI MRI — AD/NC 分類 與 MCI 轉換預測

使用 ADNI T1 結構式 MRI，以 LightCNN3D（~587K params）進行：
1. **AD vs NC 二元分類**（AUC 0.751 ± 0.056）
2. **pMCI vs sMCI 轉換預測**（AUC 0.763 ± 0.158，含臨床特徵融合）

詳細方法與結果見 [docs/final_report.md](docs/final_report.md)。

---

## 快速開始

### 環境建立

```powershell
# 建立虛擬環境（首次）
.\setup_venv.ps1

# 若 PowerShell 封鎖腳本
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 安裝 GPU 版 PyTorch（CUDA 13.0）
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-cu130.txt

# 驗證 GPU
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 執行推論（已有 Checkpoints）

```powershell
# 1. Youden's J 最佳門檻評估
.\.venv\Scripts\python.exe evaluate_youden.py --task ad_nc
.\.venv\Scripts\python.exe evaluate_youden.py --task mci_conversion --adnimerge_csv dataset/ADNIMERGE.csv

# 2. Grad-CAM 視覺化（含解剖切片）
.\.venv\Scripts\python.exe visualize_gradcam.py --task ad_nc --anatomical
.\.venv\Scripts\python.exe visualize_gradcam.py --task mci_conversion --anatomical --compare

# 3. 報告圖表
.\.venv\Scripts\python.exe plot_results.py --task light_v4
.\.venv\Scripts\python.exe plot_results.py --task mci_v2
```

### 重新訓練

```powershell
# AD vs NC（LightCNN3D v4）
.\.venv\Scripts\python.exe train_light.py \
    --data_csv data/processed_list.csv \
    --output_csv results/cv_light_v4_results.csv \
    --epochs 150 --lr 1e-4 --focal_gamma 2.0 --ad_weight 1.0 --dropout 0.2

# MCI 轉換（LightCNN3D + 臨床特徵，v3）
.\.venv\Scripts\python.exe train_mci_conversion.py \
    --data_csv data/mci_conversion_list.csv \
    --output_csv results/cv_mci_v3_results.csv \
    --adnimerge_csv dataset/ADNIMERGE.csv \
    --epochs 150 --lr 1e-4 --dropout 0.2 --pmci_weight 1.5

# MCI 轉換（純影像，v2）
.\.venv\Scripts\python.exe train_mci_conversion.py \
    --data_csv data/mci_conversion_list.csv \
    --output_csv results/cv_mci_v2_results.csv \
    --epochs 150 --lr 1e-4 --dropout 0.2 --pmci_weight 1.5
```

---

## 最終結果

### Task 1：AD vs NC（801 scans / 473 subjects）

| 方法 | AUC | Sensitivity | Specificity | 門檻 |
|------|-----|-------------|-------------|------|
| @0.5（預設） | 0.751 | 0.238 | 0.883 | 0.5 |
| **@Youden（推薦）** | **0.751** | **0.842** | **0.669** | **0.267** |

### Task 2：MCI 轉換（131 scans / 63 subjects）

#### v2（純影像）

| 方法 | AUC | Sensitivity | Specificity | 門檻 |
|------|-----|-------------|-------------|------|
| @0.5 | 0.744 | 0.771 | 0.298 | 0.5 |
| @Youden | 0.744 | 0.596 | 0.930 | 0.758 |

#### v3（影像 + 臨床特徵，推薦）

| 方法 | AUC | Sensitivity | Specificity | 門檻 |
|------|-----|-------------|-------------|------|
| @0.5 | 0.763 | 1.000 | 0.134 | 0.5 |
| **@Youden（推薦）** | **0.763** | **0.750** | **0.789** | **0.804** |

臨床特徵：AGE、PTGENDER、APOE4、MMSE_bl、CDRSB_bl（來自 ADNIMERGE）

---

## 目錄結構

```
final projecet/
├── checkpoints/
│   ├── best_light.pth          ← AD/NC 最佳模型（fold2, AUC=0.839）
│   ├── best_light_fold[1-5].pth
│   ├── best_mci.pth            ← MCI 最佳模型（fold3, AUC=0.867）
│   └── best_mci_fold[1-5].pth
│
├── data/
│   ├── processed/AD|NC|MCI/    ← 前處理後 .npy 影像
│   ├── processed_list.csv      ← AD+NC 掃描清單（801 筆）
│   └── mci_conversion_list.csv ← MCI 轉換清單（131 筆）
│
├── models/
│   ├── light_cnn3d.py          ← ⭐ 主模型（~587K params）
│   ├── baseline_cnn.py
│   └── densenet_monai.py
│
├── results/
│   ├── figures_light_v4/       ← AD/NC 訓練圖表
│   ├── figures_mci_v2/         ← MCI 訓練圖表
│   ├── gradcam_light_v4/       ← AD/NC Grad-CAM（含解剖切片）
│   ├── gradcam_baseline_v4/    ← Baseline Grad-CAM（對比）
│   ├── gradcam_mci_v3/         ← MCI Grad-CAM + pMCI vs sMCI 比較圖
│   ├── cv_light_v4_results.csv
│   ├── cv_mci_v2_results.csv
│   ├── youden_eval_adnc.csv
│   ├── youden_eval_mci.csv         ← v2 純影像
│   ├── cv_mci_v3_results.csv       ← MCI v3（臨床特徵融合）
│   └── youden_eval_mci_v3.csv      ← v3 Youden 評估
│
├── docs/
│   ├── final_report.md         ← ⭐ 完整結果報告
│   └── history_and_results.md  ← 實驗歷程紀錄
│
├── scripts/archive/            ← 已完成的一次性腳本
│
├── dataset.py                  ← 資料載入
├── train_light.py              ← ⭐ AD/NC 訓練
├── train_mci_conversion.py     ← ⭐ MCI 訓練
├── train_baseline.py           ← Baseline 訓練
├── evaluate_youden.py          ← ⭐ Youden's J 評估
├── evaluate_ensemble.py        ← Ensemble 評估
├── visualize_gradcam.py        ← ⭐ Grad-CAM 視覺化
├── plot_results.py             ← 報告圖表生成
├── augment_offline.py          ← 離線資料擴增（備用）
└── preprocess.py               ← NIFTI → .npy 前處理
```

---

## 前處理流程（從頭開始）

```powershell
# 1. 前處理 ADNI 原始影像
.\.venv\Scripts\python.exe preprocess.py \
    --input_dir dataset/ADNI \
    --metadata_csv adni_metadata.csv \
    --output_dir data

# 2. 建立 MCI 轉換標籤（需要 ADNIMERGE.csv）
.\.venv\Scripts\python.exe scripts/archive/prepare_metadata.py
.\.venv\Scripts\python.exe scripts/archive/merge_labels.py
.\.venv\Scripts\python.exe scripts/archive/prepare_mci_labels.py
.\.venv\Scripts\python.exe scripts/archive/rebuild_csv.py
```

---

## 模型架構摘要

```
LightCNN3D（~587K parameters）
Input 128³ ─→ Stem(7³,s2) ─→ Stage1(ResBlock+Down) ─→ Stage2(ResBlock+Down)
              16ch 64³          32ch 32³                64ch 16³  ← Grad-CAM
         ─→ Stage3(ResBlock+Down) ─→ GAP ─→ Dropout(0.2) ─→ FC(2)
              128ch 8³
```

ResBlock = Conv3d-BN-ReLU-Conv3d-BN + skip connection

---

## 環境需求

```
Python 3.x
torch 2.10.0+cu130
monai
numpy, pandas, scikit-learn
scipy, matplotlib, seaborn
tqdm
```

```powershell
# CPU 版
pip install -r requirements.txt

# GPU 版（本機使用）
pip install -r requirements-gpu-cu130.txt
```
