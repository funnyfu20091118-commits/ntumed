# 資料前處理流程說明

本研究採用 MIMIC-CXR-JPG 資料集，以下說明各步驟邏輯對應的程式碼位置。

---

## 1. 資料集路徑與配置

所有路徑與參數集中定義於 `Config` dataclass。

| 項目 | 檔案 | 行號 |
|------|------|------|
| 影像根目錄 `image_root` | [src/config.py](src/config.py#L14) | L14 |
| 報告根目錄 `report_root` | [src/config.py](src/config.py#L15) | L15 |
| Metadata CSV 路徑 | [src/config.py](src/config.py#L16) | L16 |
| CheXpert 標籤 CSV 路徑 | [src/config.py](src/config.py#L17) | L17 |
| Split CSV 路徑 | [src/config.py](src/config.py#L18) | L18 |
| Study 列表 CSV 路徑 | [src/config.py](src/config.py#L19) | L19 |

---

## 2. 篩選 AP / PA 視角影像

讀入 metadata 後，以 `ViewPosition` 欄位篩選僅保留 **AP** 與 **PA** 的影像，確保視角一致性。

| 項目 | 檔案 | 行號 |
|------|------|------|
| 載入 metadata / split / chexpert CSV | [src/preprocess.py](src/preprocess.py#L59-L61) | L59-61 |
| 合併 metadata 與 split 資訊 | [src/preprocess.py](src/preprocess.py#L64) | L64 |
| **篩選 AP/PA 視角** (`ViewPosition.isin(["AP","PA"])`) | [src/preprocess.py](src/preprocess.py#L67) | L67 |

---

## 3. CheXpert 14 項疾病標籤

使用 CheXpert 產出的 14 項二元標籤（含 uncertain 處理）。

| 項目 | 檔案 | 行號 |
|------|------|------|
| 14 項疾病名稱定義 (`LABEL_COLS`) | [src/dataset.py](src/dataset.py#L16-L21) | L16-21 |
| 合併 CheXpert 標籤至 manifest | [src/preprocess.py](src/preprocess.py#L120) | L120 |
| 讀取標籤 & uncertain (-1) 處理 | [src/dataset.py](src/dataset.py#L49-L62) | L49-62 |
| `num_labels = 14` 設定 | [src/config.py](src/config.py#L52) | L52 |

**Uncertain 處理邏輯**（[src/dataset.py](src/dataset.py#L56-L59)）：CheXpert 中值為 `-1`（uncertain）的標籤，label 設為 `0.0`，mask 設為 `0.0`（訓練時忽略）。

---

## 4. 報告文字清洗（Findings + Impression）

從原始報告檔案中擷取 **FINDINGS** 與 **IMPRESSION** 段落，並進行文字清洗。

| 項目 | 檔案 | 行號 |
|------|------|------|
| `parse_report()` 函式定義 | [src/preprocess.py](src/preprocess.py#L19) | L19 |
| 以正則匹配 section header（全大寫） | [src/preprocess.py](src/preprocess.py#L28) | L28 |
| 優先取 FINDINGS + IMPRESSION 段落 | [src/preprocess.py](src/preprocess.py#L37-L41) | L37-41 |
| 若無標準段落，fallback 使用全文 | [src/preprocess.py](src/preprocess.py#L42-L44) | L42-44 |
| 清除多餘空白 (`\s+`) | [src/preprocess.py](src/preprocess.py#L47) | L47 |
| 清除去識別化符號 (`___`) | [src/preprocess.py](src/preprocess.py#L48) | L48 |
| 批次解析所有報告 | [src/preprocess.py](src/preprocess.py#L107-L113) | L107-113 |
| 過濾空報告（長度 ≤ 10） | [src/preprocess.py](src/preprocess.py#L116) | L116 |

---

## 5. 建立圖文對應 Manifest

將影像路徑、報告文字、CheXpert 標籤整合為單一 CSV manifest。

| 項目 | 檔案 | 行號 |
|------|------|------|
| 組合影像路徑 `make_img_path()` | [src/preprocess.py](src/preprocess.py#L70-L75) | L70-75 |
| 組合報告路徑 `make_report_path()` | [src/preprocess.py](src/preprocess.py#L79-L84) | L79-84 |
| 建立磁碟影像索引（`find *.jpg`） | [src/preprocess.py](src/preprocess.py#L88-L93) | L88-93 |
| 檢查影像與報告是否實際存在 | [src/preprocess.py](src/preprocess.py#L100-L104) | L100-104 |
| 去重：同 study 保留 PA 優先於 AP | [src/preprocess.py](src/preprocess.py#L108-L113) | L108-113 |
| 儲存 manifest CSV 至 `cache/manifest.csv` | [src/preprocess.py](src/preprocess.py#L127-L133) | L127-133 |

Manifest 欄位：`dicom_id, subject_id, study_id, split, image_path, report_path, report_text, ViewPosition` + 14 項 CheXpert 標籤。

---

## 6. Dataset 載入（訓練時）

訓練時透過 `MIMICCXRDataset` 讀取 manifest，回傳 `(image, report, labels, label_mask)` 四元組。

| 項目 | 檔案 | 行號 |
|------|------|------|
| `MIMICCXRDataset` 類別 | [src/dataset.py](src/dataset.py#L13) | L13 |
| 依 split 篩選 manifest | [src/dataset.py](src/dataset.py#L25) | L25 |
| 影像前處理 (Resize → Tensor → Normalize [-1,1]) | [src/dataset.py](src/dataset.py#L28-L32) | L28-32 |
| `__getitem__` 回傳資料 | [src/dataset.py](src/dataset.py#L38-L64) | L38-64 |
