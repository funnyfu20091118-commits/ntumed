# Chest-Diffusion 條件機制：程式碼對照表

## 1. Full Report（全篇報告）→ 文字向量 (Text Token)

| 步驟 | 檔案 | 行號 | 說明 |
|------|------|------|------|
| 資料集回傳 report 文字 | [src/dataset.py](src/dataset.py#L46) | L46 | `report = str(row["report_text"])` 從 manifest 讀取醫師報告原文 |
| Tokenize + CLIP encode | [src/train_uvit.py](src/train_uvit.py#L63-L68) | L63-68 | `encode_text()` 用 BiomedCLIP tokenizer 把文字轉 token，再用 `clip_model.encode_text()` 產生 512 維向量 |
| BiomedCLIP 微調（Stage 1） | [src/train_clip.py](src/train_clip.py#L36-L43) | L36-43 | 在 MIMIC-CXR 上用 contrastive loss 微調 BiomedCLIP，使其更適應胸腔 X 光報告 |
| TextProjection 投影到 U-ViT 維度 | [src/uvit.py](src/uvit.py#L103-L110) | L103-110 | `TextProjection` 類別：`nn.Linear(512→512) + LayerNorm`，將 CLIP 輸出投影到 U-ViT 的 token 維度 |
| forward 中產生 text_token | [src/uvit.py](src/uvit.py#L221-L222) | L221-222 | `text_token = self.text_proj(text_emb).unsqueeze(1)` → shape `(B, 1, dim)` |

---

## 2. Meta（14 個 CheXpert 標籤）→ 標籤向量 (Label Token)

| 步驟 | 檔案 | 行號 | 說明 |
|------|------|------|------|
| 14 個標籤定義 | [src/dataset.py](src/dataset.py#L17-L21) | L17-21 | `LABEL_COLS`：Atelectasis, Cardiomegaly, Consolidation, Edema, … 共 14 種 |
| 資料集回傳 labels tensor | [src/dataset.py](src/dataset.py#L49-L63) | L49-63 | 從 manifest 讀取每個標籤值，處理 NaN / uncertain (-1)，回傳 `(14,)` float tensor |
| LabelProjection 類別 | [src/uvit.py](src/uvit.py#L113-L122) | L113-122 | `nn.Linear(14→512) → SiLU → nn.Linear(512→512) → LayerNorm`，把 14 維標籤投影成一個 512 維 token |
| forward 中產生 label_token | [src/uvit.py](src/uvit.py#L224-L227) | L224-227 | `label_token = self.label_proj(labels).unsqueeze(1)` → shape `(B, 1, dim)` |
| num_labels 設定 | [src/config.py](src/config.py#L60) | L60 | `num_labels: int = 14` |

---

## 3. U-ViT 中的 Concatenation（全部連在一起）

| 步驟 | 檔案 | 行號 | 說明 |
|------|------|------|------|
| Time token | [src/uvit.py](src/uvit.py#L219) | L219 | `time_token = self.time_embed(t).unsqueeze(1)` → `(B, 1, dim)` |
| Image patch tokens | [src/uvit.py](src/uvit.py#L216) | L216 | `img_tokens = self.patch_embed(z_t)` → `(B, 256, dim)`（32/2=16, 16×16=256 patches） |
| Text token | [src/uvit.py](src/uvit.py#L222) | L222 | `(B, 1, dim)` |
| Label token | [src/uvit.py](src/uvit.py#L225) | L225 | `(B, 1, dim)` |
| **Concatenate** | [src/uvit.py](src/uvit.py#L230) | **L230** | `tokens = torch.cat([time_token, img_tokens, text_token, label_token], dim=1)` |
| Positional embedding | [src/uvit.py](src/uvit.py#L233) | L233 | `tokens = tokens + self.pos_embed`，total tokens = 1 + 256 + 1 + 1 = **259** |
| 進入 Encoder → Middle → Decoder | [src/uvit.py](src/uvit.py#L236-L249) | L236-249 | 8 encoder blocks + 1 middle block + 8 decoder blocks（含 skip connections） |
| 取出 image tokens → 還原空間 | [src/uvit.py](src/uvit.py#L252-L255) | L252-255 | `img_tokens = tokens[:, 1:1+num_patches, :]` 跳過 time token，再 unpatch 回 `(B, C, H, W)` |

---

## 整體流程圖（對應程式碼）

```
醫師報告 (str)                         14 個 CheXpert 標籤 (float[14])
    │                                         │
    ▼                                         ▼
BiomedCLIP.encode_text()              LabelProjection (uvit.py L113-122)
  (train_uvit.py L63-68)                14 → 512 → 512
    │                                         │
    ▼                                         ▼
TextProjection (uvit.py L103-110)     label_token (B,1,512)
  512 → 512 + LayerNorm                      │
    │                                         │
    ▼                                         │
text_token (B,1,512)                          │
    │         ┌─────────────┐                 │
    │         │ TimestepEmb │                 │
    │         │ (uvit.py    │                 │
    │         │  L19-34)    │                 │
    │         └──────┬──────┘                 │
    │           time_token                    │
    │           (B,1,512)                     │
    │                │                        │
    │    PatchEmbed  │                        │
    │   (uvit.py     │                        │
    │    L38-49)     │                        │
    │  img_tokens    │                        │
    │  (B,256,512)   │                        │
    │       │        │                        │
    ▼       ▼        ▼                        ▼
   ┌─────────────────────────────────────────────┐
   │  torch.cat(...)        (uvit.py L230)       │
   │  → (B, 259, 512)                            │
   │  + pos_embed           (uvit.py L233)       │
   └────────────────────┬────────────────────────┘
                        ▼
              Encoder (8 blocks) L236-238
                        ▼
              Middle (1 block)  L241
                        ▼
              Decoder (8 blocks + skip) L244-249
                        ▼
              取出 image tokens → UnPatch → ε(B,C,H,W)
```
