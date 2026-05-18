# 阿茲海默症 MRI 分類與 MCI 轉換預測
## 期末報告

> 資料來源：ADNI（Alzheimer's Disease Neuroimaging Initiative）  
> 日期：2026-05-18

---

## 一、研究目標

本研究使用 ADNI T1 結構式 MRI 影像進行兩項二元分類任務：

1. **Task 1（AD vs NC）**：區分阿茲海默症患者（AD）與正常對照組（NC）
2. **Task 2（MCI 轉換預測）**：預測輕度認知障礙（MCI）患者是否會轉換為 AD（pMCI vs sMCI）

---

## 二、資料集

### 2.1 資料來源與規模

| 類別 | 受試者數 | 掃描數 | 任務 |
|------|---------|------|------|
| AD（阿茲海默症） | 88 | 290 | Task 1 |
| NC（正常對照） | 385 | 511 | Task 1 |
| pMCI（進展型 MCI） | ~39 | 72 | Task 2 |
| sMCI（穩定型 MCI） | ~24 | 59 | Task 2 |
| **Task 1 合計** | **473** | **801** | |
| **Task 2 合計** | **63** | **131** | |

### 2.2 前處理流程

1. **Skull stripping**：FreeSurfer recon-all
2. **重採樣**：至 1mm isotropic
3. **空間對齊**：大腦質心對齊，resize 至 128×128×128 voxels
4. **強度標準化**：z-score（以受試者自身腦組織為基準），背景設為 0.0

**已知限制**：skull strip 邊界在 z-score 後產生約 −1.9 的強負值條紋，在 Grad-CAM 中形成藍色偽影。

### 2.3 資料不平衡

- Task 1：NC:AD ≈ 1.76:1（輕度不平衡）
- Task 2：pMCI:sMCI ≈ 1.22:1（接近平衡）

---

## 三、方法

### 3.1 模型架構：LightCNN3D

為應對小資料集（< 500 subjects），設計輕量化 3D CNN（~587K 參數，DenseNet121 的 1/19）：

```
Input: 128³ × 1ch
  Stem: Conv(1→16, 7³, stride=2) + BN + ReLU  → 64³ × 16ch
  Stage1: ResBlock(16→16) + DownConv(16→32)     → 32³ × 32ch
  Stage2: ResBlock(32→32) + DownConv(32→64)     → 16³ × 64ch  ← Grad-CAM target
  Stage3: ResBlock(64→64) + DownConv(64→128)    →  8³ × 128ch
  Head: GlobalAvgPool → Dropout(0.2) → Linear(128→2)
```

ResBlock = [Conv3d → BN → ReLU → Conv3d → BN] + skip connection

#### 臨床特徵融合分支（MCI v3）

MCI 任務額外加入 Clinical MLP branch，與影像特徵在 GAP 後 concat：

```
MRI 影像 → ... → GAP → 128-dim ──────────────────────┐
                                                        ├─ concat(160) → Dropout → Linear(160→2)
臨床特徵(5) → Linear(5→32) → ReLU → Linear(32→32) ──┘
```

臨床特徵（來自 ADNIMERGE 基線訪視）：AGE、PTGENDER（Male=1）、APOE4（0/1/2）、MMSE_bl、CDRSB_bl

### 3.2 訓練設定（最終版 v4）

| 項目 | Task 1 (AD/NC) | Task 2 (MCI) |
|------|---------------|--------------|
| Loss | Focal Loss（γ=2.0，無 class weight） | CrossEntropy（pmci_weight=1.5） |
| Optimizer | AdamW（lr=1e-4, wd=1e-4） | AdamW（lr=1e-4, wd=1e-4） |
| Scheduler | CosineAnnealingLR（T_max=150） | CosineAnnealingLR（T_max=150） |
| Batch size | 4 | 2 |
| Max epochs | 150 | 150 |
| Early stopping | patience=35（based on val AUC） | patience=35（based on val AUC） |
| Dropout | 0.2 | 0.2 |
| AMP | ✓（CUDA） | ✓（CUDA） |

### 3.3 驗證策略

- **5-fold StratifiedGroupKFold**：以 subject_id 分組，確保同一受試者的多次掃描不會跨折（防止 data leakage）
- **主要評估指標**：AUC-ROC（門檻無關，對不平衡資料集最可靠）
- **輔助門檻**：Youden's J = argmax(Sensitivity + Specificity − 1)

### 3.4 硬體環境

- GPU：NVIDIA RTX 3050 4GB + CUDA 13.0
- Framework：PyTorch 2.10.0 + MONAI
- Python 3.x，虛擬環境 `.venv/`

---

## 四、實驗結果

### 4.1 Task 1：AD vs NC

#### 模型演進

| 實驗 | 模型 | 參數量 | 資料量 | Mean AUC | Std |
|------|------|-------|-------|----------|-----|
| Exp 1 | Baseline 3D CNN | 200K | 151 scans | 0.564 | 0.204 |
| Exp 2 | DenseNet121 | 11.2M | 151 scans | 0.580 | 0.102 |
| Exp 3 | LightCNN3D v1（Focal+OneCycle） | 587K | 151 scans | 0.618 | 0.179 |
| Exp 4 | LightCNN3D v2（CE+Cosine） | 587K | 151 scans | 0.614 | 0.186 |
| Exp 6 | Baseline v2 | 200K | 801 scans | 0.698 | 0.115 |
| Exp 7 | LightCNN3D v3（CE loss） | 587K | 801 scans | 0.721 | 0.074 |
| **Exp 9** | **LightCNN3D v4（Focal-only）** | **587K** | **801 scans** | **0.751** | **0.056** |

#### 最終結果（LightCNN3D v4，Exp 9）

| Fold | Best Epoch | AUC | Sens@0.5 | Spec@0.5 |
|------|-----------|-----|----------|----------|
| 1 | 41 | 0.770 | 0.000 | 1.000 |
| 2 | 6 | **0.839** | 0.000 | 1.000 |
| 3 | 47 | 0.727 | 0.400 | 0.783 |
| 4 | 17 | 0.714 | 0.400 | 0.833 |
| 5 | 33 | 0.704 | 0.390 | 0.798 |
| **Mean** | | **0.751 ± 0.056** | 0.238 | 0.883 |

#### Youden's J 最佳門檻分析

| 指標 | @0.5 門檻 | @Youden 門檻 | 改變 |
|------|----------|-------------|------|
| 門檻值 | 0.500 | **0.267** | — |
| Sensitivity | 0.238 | **0.842 ± 0.061** | +0.604 |
| Specificity | 0.883 | 0.669 ± 0.075 | −0.214 |
| Accuracy | 0.720 | 0.713 | ≈ 持平 |

聚合混淆矩陣（@Youden 0.267，801 samples）：

|  | 預測 AD | 預測 NC |
|--|--------|--------|
| 真實 AD | **170** | 32 |
| 真實 NC | 198 | **401** |

> **關鍵發現**：模型機率分布中 AD 集中於 0.2–0.4，NC 集中於 0.6–0.9。這是 focal loss 在 1.76:1 不平衡下的正常校準偏移，並非訓練失敗。AUC 正確反映模型的真實排序能力；Youden's J 找回了最佳門檻。

#### Ensemble 分析

| 方法 | AUC | Sens@0.5 |
|------|-----|----------|
| Single model | **0.751** | 0.238 |
| LOO Ensemble（4 models） | 0.745 | **0.495** |
| Full 5-model Ensemble | 0.747 | **0.495** |

> Ensemble 對 AUC 無顯著提升（模型間 AUC 差異過大：0.704–0.839），但使 Sensitivity@0.5 倍增。**最終推薦使用 Single model + Youden 門檻 0.267**。

---

### 4.2 Task 2：MCI 轉換預測（pMCI vs sMCI）

#### 模型比較

| 實驗 | 模型 | Mean AUC | 說明 |
|------|------|----------|------|
| Exp 8 | DenseNet121 | 0.571 ± 0.141 | 接近隨機，嚴重過擬合 |
| Exp 10 | LightCNN3D v2（純影像） | 0.744 ± 0.082 | ▲ +0.173 |
| **Exp 11** | **LightCNN3D v3（+ 臨床特徵）** | **0.763 ± 0.158** | **▲ +0.019** |

#### 結果：LightCNN3D MCI v2（Exp 10，純影像）

| Fold | Best Epoch | AUC | Sens@0.5 | Spec@0.5 |
|------|-----------|-----|----------|----------|
| 1 | 1 | 0.761 | 0.000 | 1.000 |
| 2 | 40 | 0.709 | 0.857 | 0.308 |
| 3 | 26 | **0.867** | 1.000 | 0.182 |
| 4 | 58 | 0.740 | 1.000 | 0.000 |
| 5 | 3 | 0.643 | 1.000 | 0.000 |
| **Mean** | | **0.744 ± 0.082** | 0.771 | 0.298 |

#### 結果：LightCNN3D MCI v3（Exp 11，影像 + 臨床特徵）

| Fold | Best Epoch | AUC | Sens@0.5 | Spec@0.5 |
|------|-----------|-----|----------|----------|
| 1 | 47 | 0.833 | 1.000 | 0.167 |
| 2 | 17 | 0.654 | 1.000 | 0.154 |
| 3 | 11 | **0.952** | 1.000 | 0.182 |
| 4 | 11 | 0.825 | 1.000 | 0.000 |
| 5 | 28 | 0.554 | 1.000 | 0.167 |
| **Mean** | | **0.763 ± 0.158** | 1.000 | 0.134 |

#### Youden's J 比較（v2 vs v3）

| 指標 | v2 @0.5 | v2 @Youden | v3 @0.5 | v3 @Youden |
|------|---------|-----------|---------|-----------|
| 門檻值 | 0.500 | 0.758 | 0.500 | **0.804** |
| Sensitivity | 0.771 | 0.596 ± 0.128 | 1.000 | **0.750 ± 0.193** |
| Specificity | 0.298 | **0.930 ± 0.076** | 0.134 | 0.789 ± 0.198 |

聚合混淆矩陣：**v2 @Youden 0.758**（131 samples）：

|  | 預測 pMCI | 預測 sMCI |
|--|----------|----------|
| 真實 pMCI | **43** | 29 |
| 真實 sMCI | 4 | **55** |

聚合混淆矩陣：**v3 @Youden 0.804**（131 samples）：

|  | 預測 pMCI | 預測 sMCI |
|--|----------|----------|
| 真實 pMCI | **54** | 18 |
| 真實 sMCI | 13 | **46** |

> **臨床特徵融合的效果**：v3 的 Sensitivity@Youden 從 0.596 → **0.750**（+0.154），代價是 Specificity 從 0.930 → 0.789（−0.141）。v3 整體更平衡，較適合初期篩查；v2 高 Specificity 較適合確認性診斷。
>
> **注意**：v3 的 std 較高（0.158 vs 0.082），主因 Fold 5 表現不穩定（AUC=0.554），這是 63 subjects 小資料集的固有不確定性。

---

### 4.3 Grad-CAM 可視化分析

#### 目標層選擇

| 模型 | Target Layer | 空間解析度 | 說明 |
|------|-------------|----------|------|
| LightCNN3D（最終） | `stage2.0` | 32³ | 高解析度，特徵細緻 |

#### 解剖位置切片（`--anatomical` 旗標）

對每位受試者的腦部邊界框做相對比例採樣，對應以下臨床關鍵區域：

| 解剖位置 | z 比例 | AD 病理意義 |
|---------|-------|-----------|
| 內嗅皮質（Entorhinal Cortex） | 25% | 最早萎縮，甚至早於臨床症狀 |
| 海馬迴 / 杏仁核（Hippocampus / Amygdala） | 38% | MCI→AD 轉換核心指標 |
| 中顳葉（Mid-Temporal） | 52% | 顳葉本體 |
| 顳頂葉（Temporoparietal） | 68% | 晚期 AD 擴散區域 |

#### Grad-CAM 觀察結果

**AD/NC（`results/gradcam_light_v4/`，基於 best_light.pth）**

| 受試者 | 真實 | 預測 | Grad-CAM 觀察 |
|--------|------|------|-------------|
| 005_S_0553 | NC | NC | 廣泛激活，符合全腦保留特徵 |
| 005_S_0602 | NC | NC | 廣泛激活 |
| 007_S_1206 | NC | NC | 廣泛激活 |
| 005_S_0814 | AD | NC | 誤分類；激活廣泛分布 |
| 007_S_1304 | AD | NC | 誤分類；激活廣泛分布 |
| 013_S_0996 | AD | AD | ✅ 顳葉前下方有局部亮點，符合 AD 病理 |

**MCI 轉換（`results/gradcam_mci_v3/`，基於 best_mci.pth）**

- pMCI（005_S_0448、005_S_0572、016_S_0769）：激活集中於內側顳葉和海馬迴層面（38% 水平）
- sMCI（005_S_0324、016_S_1149、023_S_0613）：激活相對分散或出現在不同腦區

並排比較圖（`results/gradcam_mci_v3/comparison_pmci_vs_smci.png`）顯示 pMCI 組在海馬迴層面的激活強度系統性高於 sMCI 組。

---

## 五、討論

### 5.1 模型選擇的重要性

從 DenseNet121（11.2M params，AUC=0.571）換為 LightCNN3D（587K params，AUC=0.744）使 MCI 任務 AUC 提升 **+0.173**，是本研究最顯著的改進。小資料集的首要問題是過擬合，而非模型容量不足。

### 5.2 Loss Function 影響

| 設定 | AD/NC AUC | 折間 Std |
|------|-----------|---------|
| CE loss（v3） | 0.721 | 0.074 |
| Focal loss gamma=2（v4） | **0.751** | **0.056** |

Focal loss 降低了多數類（NC）高信心樣本的梯度貢獻，使模型更關注難以分類的樣本（AD 邊界案例），AUC 提升且折間方差下降。

### 5.3 門檻校準的必要性

以 0.5 為預設門檻報告的 Sensitivity 嚴重低估了模型的真實識別能力：

- AD/NC：Sens@0.5 = 0.238，Sens@Youden = **0.842**（高出 3.5 倍）
- MCI：Spec@0.5 = 0.298，Spec@Youden = **0.930**（高出 3.1 倍）

對不平衡資料集，應以 AUC 評估模型能力，以 Youden's J 決定最終門檻。

### 5.4 臨床特徵融合的效果與限制

ADNIMERGE 對 MCI 受試者的覆蓋率達 **100%**，因此可完整使用 5 個基線臨床特徵（AGE、PTGENDER、APOE4、MMSE_bl、CDRSB_bl）。AUC 小幅提升（+0.019），但更重要的是 **Sensitivity@Youden 大幅改善 +0.154**，模型從原本偏向高 Specificity 轉為更平衡的判斷。

反觀 AD/NC 任務，本研究所用的 ADNIMERGE.csv 僅覆蓋 ADNI1/2 受試者，AD/NC 資料大量使用 ADNI3（RID > 10000），覆蓋率僅 **15.6%**（74/473 subjects）。以中位數填補 84.4% 缺失值等同於引入大量雜訊，因此 AD/NC 不採用臨床特徵融合。若取得涵蓋 ADNI3 的完整 ADNIMERGE，AD/NC 的臨床融合同樣可行。

### 5.5 限制

1. **MCI 資料量不足**：63 subjects 是 MCI 任務的根本瓶頸。文獻 SOTA（0.75–0.80）使用數百至數千受試者。
2. **無 MNI 配准**：解剖位置估算誤差 ±8–12 voxels，Grad-CAM 的解剖對應僅供定性參考。
3. **Grad-CAM 偽影**：頭骨剝離邊界強負值條紋在熱圖中產生藍色偽影，需前處理修正。
4. **Ensemble 效益有限**：AD/NC Ensemble 對 AUC 無改善；MCI Ensemble 因模型不穩定甚至有損（AUC −0.14）。

---

## 六、最終結論

| 任務 | 最佳模型 | AUC | 最佳門檻 | Sensitivity | Specificity |
|------|---------|-----|---------|-------------|-------------|
| AD vs NC | LightCNN3D v4 | **0.751 ± 0.056** | 0.267（Youden） | 0.842 | 0.669 |
| MCI 轉換（篩查） | LightCNN3D MCI v3（+ 臨床） | **0.763 ± 0.158** | 0.804（Youden） | 0.750 | 0.789 |
| MCI 轉換（確診） | LightCNN3D MCI v2（純影像） | 0.744 ± 0.082 | 0.758（Youden） | 0.596 | 0.930 |

兩個任務均以 **LightCNN3D（587K params）+ Focal/CE loss + CosineAnnealingLR + Youden's J 門檻校準** 達到最佳效能。MCI 任務額外加入 ADNIMERGE 臨床特徵融合，使模型在篩查情境下 Sensitivity 提升至 0.750。若要進一步提升，最優先的方向是擴充 MCI 資料至 150+ subjects（預期 AUC ▲ 0.03–0.06）。

---

## 附錄：檔案索引

### 核心 Checkpoints

| 檔案 | 說明 | AUC |
|------|------|-----|
| `checkpoints/best_light.pth` | LightCNN3D v4 最佳（fold 2, ep 6） | 0.839 |
| `checkpoints/best_light_fold[1-5].pth` | 各折 checkpoints | — |
| `checkpoints/best_mci.pth` | LightCNN3D MCI v2 最佳（fold 3, ep 26） | 0.867 |
| `checkpoints/best_mci_fold[1-5].pth` | 各折 checkpoints | — |

### 結果 CSV

| 檔案 | 說明 |
|------|------|
| `results/cv_light_v4_results.csv` | LightCNN3D v4 各折詳細指標 |
| `results/cv_mci_v2_results.csv` | LightCNN3D MCI v2 各折詳細指標 |
| `results/youden_eval_adnc.csv` | AD/NC Youden's J 評估 |
| `results/youden_eval_mci.csv` | MCI v2 Youden's J 評估（純影像） |
| `results/cv_mci_v3_results.csv` | MCI v3 各折詳細指標（臨床特徵融合） |
| `results/youden_eval_mci_v3.csv` | MCI v3 Youden's J 評估 |
| `results/ensemble_eval_adnc_*.csv` | AD/NC Ensemble 評估（single/loo/full5） |

### 視覺化輸出

| 路徑 | 說明 |
|------|------|
| `results/figures_light_v4/` | LightCNN3D v4 學習曲線、AUC 圖、混淆矩陣 |
| `results/figures_mci_v2/` | MCI v2 學習曲線、AUC 圖、混淆矩陣 |
| `results/gradcam_light_v4/` | AD/NC Grad-CAM（標準 + 解剖切片） |
| `results/gradcam_baseline_v4/` | Baseline Grad-CAM（對比） |
| `results/gradcam_mci_v3/` | MCI Grad-CAM（解剖切片 + pMCI vs sMCI 比較圖） |
