# preprocess.py 功能與用法說明

本文件說明 [src/preprocess.py](src/preprocess.py) 的目的、處理流程、輸入輸出，以及如何執行。

## 1. 這支程式在做什麼？

[src/preprocess.py](src/preprocess.py) 會建立訓練用的資料清單（manifest），把每筆胸腔 X 光影像對應到報告文字與 CheXpert 標籤，最後輸出到 `cache/manifest.csv`。

主要目標：
- 合併 metadata 與 split（train/validate/test）
- 只保留 AP / PA 影像
- 檢查影像與報告是否存在
- 每個 study 去重（同 study 優先保留 PA）
- 解析報告文字（優先 FINDINGS + IMPRESSION）
- 合併 CheXpert 標籤
- 匯出最終 manifest

---

## 2. 核心函式

### `parse_report(report_path)`
用途：讀取單一報告檔，抽出可用文字。

處理邏輯：
- 讀取 `.txt` 報告
- 用全大寫 section header 切段（例如 FINDINGS、IMPRESSION）
- 優先拼接 FINDINGS 與 IMPRESSION
- 若找不到標準段落，改用全文
- 清理空白字元與去識別底線（例如 `___`）

回傳值：
- 一段乾淨的 report 文字（`str`）

### `build_manifest(cfg)`
用途：建立完整資料表（DataFrame）並寫出 `manifest.csv`。

主要步驟：
1. 讀入 metadata、split、chexpert CSV。
2. 依 `dicom_id` 合併 split。
3. 篩選 `ViewPosition` 只留 AP / PA。
4. 依規則組出 `image_path` 與 `report_path`。
5. 先掃描磁碟上的 jpg，建立 existing image set。
6. 以 `study_csv` 建立 existing study set。
7. 保留「影像存在 + study 存在」的資料列。
8. 以 `(subject_id, study_id)` 去重，PA 優先於 AP。
9. 解析 report 文字，並過濾過短內容。
10. 合併 CheXpert 標籤。
11. 輸出 `cache/manifest.csv`。

---

## 3. 需要的輸入資料

路徑由 [src/config.py](src/config.py) 的 `Config` 提供，關鍵欄位如下：
- `image_root`
- `report_root`
- `metadata_csv`
- `split_csv`
- `chexpert_csv`
- `study_csv`
- `cache_dir`

只要這些路徑正確，`preprocess.py` 就能直接執行。

---

## 4. 如何執行

在專案根目錄執行：

```bash
python src/preprocess.py
```

建議先確認依賴套件：

```bash
pip install pandas tqdm
```

執行過程會印出：
- AP/PA 篩選後筆數
- 檔案存在性檢查後筆數
- 去重後筆數
- 各 split（train / validate / test）統計
- manifest 輸出路徑

---

## 5. 輸出檔案

### `cache/manifest.csv`
最終欄位包含：
- 基本識別：`dicom_id`, `subject_id`, `study_id`, `split`
- 路徑：`image_path`, `report_path`
- 文字：`report_text`
- 視角：`ViewPosition`
- 標籤：CheXpert 各疾病欄位（由 `chexpert_csv` 動態加入）

另會產生：
- `cache/img_list.txt`：掃描到的 jpg 路徑索引

---

## 6. 注意事項

- 此程式使用 `find` 指令掃描影像，預設適合 Linux/macOS 環境。
- `study_csv` 與影像目錄版本要一致，否則存在性檢查會大量過濾資料。
- 匯入 `config` 採用同目錄模組方式，請用 `python src/preprocess.py` 從專案根目錄執行。

---

## 7. 快速檢查結果

執行完成後可用下列方式快速確認：

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv('cache/manifest.csv')
print(df.shape)
print(df['split'].value_counts())
print(df[['image_path', 'report_text']].head(3))
PY
```

如果上述能正常輸出，表示資料前處理流程已完成。