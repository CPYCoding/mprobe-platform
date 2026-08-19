# Dataset Verification — 設計規格文件

> 這份文件給一個全新、沒有任何前置對話記憶的 Claude Code session 看，目的是讓它能獨立照著這份文件把功能建出來，不需要額外問背景。請完整讀完再開始動工。

## 一、這是什麼專案

**MProbe** 是一個 ML 模型/資料集的安全認證市集平台。這個 repo（`mprobe-platform`）目前主要是靜態前端原型：

```
mprobe-platform/
├── marketplace.html   # 主要商店頁面：瀏覽、篩選、購物車、賣家/買家個人頁
├── upload.html        # 賣家「上架模型/資料集」的流程頁面（目前是假的模擬掃描，沒有真的邏輯）
├── models/*.html       # 各個模型的「Run（推論）」頁面
└── server/             # 之前寫過一版 dataset-cleaning 的後端實作（見下方「注意」）
```

`marketplace.html` 上有一段文案早就寫好但沒有真正實作："Upload → MProbe runs an integrity & poison scan → it goes live once certified"（上傳 → MProbe 做完整性/毒素掃描 → 通過認證才上架）。**這次要做的事，就是把這段文案變成真的**。

**⚠️ 注意：`server/` 資料夾裡已經有一版舊的實作（`materialized_trigger.py`、`blind_feature.py`、`pipeline.py` 等），那是之前另一個方向做的東西，這次要整個重新開始，不需要延續、不需要保留相容性、不需要參考裡面的設計。可以直接視為要重寫的對象，看到能重用的基礎建設（例如 FastAPI 的 zip 上傳處理）可以借鏡，但偵測邏輯本身要照這份文件的第三節，從另一個 repo 搬。**

## 二、要做的功能：資料集驗證關卡（不是市集商品）

**這不是使用者會去買、去點擊執行的商品。** 是賣家上傳資料集到平台時，系統自動執行的驗證流程：

```
賣家上傳資料集（ImageFolder 格式 zip：一個資料夾一個類別，裡面放圖片）
  → 狀態：未驗證（unverified）
  → 平台背景自動跑 poison detection 掃描（見第三節的三支偵測器）
  → 掃描完成，依結果決定
  → 狀態：已驗證（verified）／需要複審／其他（判定標準見第五節，要在實作時定案）
```

買家在 `marketplace.html` 瀏覽資料集列表時，應該看得到每筆資料集是否已驗證（verified/unverified 狀態標示）。

`models/dataset-cleaning.html` 這個獨立頁面**不需要存在**——這個功能不是一個買家會點進去單獨執行的「Run」頁面，是上傳流程背後的一道關卡。

## 三、偵測邏輯：從 `secureML-demo` 這個 repo 原封不動搬過來

偵測演算法的來源是另一個 GitHub repo：

```
https://github.com/aurelia0029/secureML-demo/tree/main/data_washing_experiment
```

裡面有三支偵測器腳本，**演算法邏輯要照抄，不要重新設計或簡化**：

1. **`materialized_trigger_detector.py`** — 已知觸發器比對。算每張圖在觸發器位置的像素跟已知觸發器圖案（`BACKDOOR_PATTERN`，定義在 `clean_reference_detector.py` 裡）的 MSE 距離，用一個從乾淨參照資料算出來的分位數門檻（`--threshold-quantile`，原始預設 0.01）判斷是否命中。

2. **`blind_feature_detector.py`** — 特徵空間異常偵測。訓練一個 `SmallCifarClassifier`（定義在 `clean_reference_detector.py` 裡，一個中等大小的 CNN），對每張圖算：離自己預測類別 centroid 的距離（`pred_centroid_dist`）、離真實類別 centroid 的距離（`true_centroid_dist`）、離最近 centroid 的距離（`nearest_centroid_dist`）、預測信心（`confidence`）、機率差距（`prob_margin`）。這些是多欄位診斷輸出，**不要合併簡化成單一分數**，照原始格式保留。

3. **`clean_reference_detector.py`** — 觸發已知圖案測信心變化。訓練一個分類器後，對每張圖分別算「不貼觸發器」跟「貼上已知觸發器」時，模型對 backdoor 目標類別（`--backdoor-label`，原始預設 9）的預測機率各是多少，兩者的差（`trigger_delta`）當作分數——差距很小，代表這張圖可能已經被訓練成看到這個圖案就會被誤判成目標類別，也就是可能已經中毒。

這三支互相有共用依賴（`SmallCifarClassifier`、`BACKDOOR_PATTERN`/`BACKDOOR_X_TOP`/`BACKDOOR_Y_TOP`/`MASK_VALUE`、`apply_trigger`/`build_trigger_tensors`、`CIFAR10_MEAN`/`CIFAR10_STD`），都定義在 `clean_reference_detector.py` 裡，被另外兩支 import。

**不要搬 `poison_audit.py`**——那支深度綁定另一個特定聯邦學習框架的物件（`hlpr.task`、`hlpr.params`、`getClientDataIndex`），不是一支可以獨立對一份資料集執行的腳本，用不上。

## 四、原始腳本 vs 這次要做的功能，有兩處落差需要調整

原始三支腳本是**研究用的評測腳本**，設計上假設兩個東西存在：

1. **聯邦學習的「client」結構**——從 `all_client.txt`/`log.txt` 解析出「哪些 client 好、哪些可疑」，再分別跑不同 client 的資料。
2. **`ground_truth_poison_indices.txt`**——已知哪些樣本真的被下毒，拿來算 precision/recall/F1。

**這次的情境是：賣家上傳一份全新資料集，平台事先不知道答案，也沒有「client」這個概念（一個賣家對應一份資料）。** 這兩樣東西在正式功能裡不會存在，所以搬過來的時候：

- **拿掉「跟 ground truth 比對算 precision/recall/F1」的部分**——正式環境沒有答案可以比對，只需要回報「哪些樣本被標記、每個偵測器給的診斷數值是多少」，不需要算準確率指標。
- **拿掉「先讀 client 好壞名單」的部分**——直接對賣家上傳的整包資料跑，不切 client，全部樣本都當同一批處理。

**除了這兩點之外，偵測演算法本身（MSE 距離計算、centroid 距離、trigger-delta 信心變化、訓練流程）要維持原樣，不要重新設計。**

## 五、已知、刻意接受的風險：現場訓練不穩定

`blind_feature_detector.py`、`clean_reference_detector.py` 原本的行為是**每次執行都從零訓練一個 `SmallCifarClassifier`**（原始預設 5 epochs），不是離線訓練一次存檔、上線後只做推論。

這個「每次現場訓練」的行為，實測過會不穩定：同樣程式碼、同樣資料規模，只換一個隨機種子，訓練出來的準確率可以有很大幅度的波動（曾經測出從 0% 到 63%+ 都有）。**這次先照原始行為保留現場訓練，暫不處理這個不穩定性**，之後有需要再另外決定要不要改成離線訓練。

**實務面提醒**：因為訓練會發生在賣家上傳資料集的當下，訓練這件事本身可能要跑上好幾分鐘（CPU 環境下，5 epochs 訓練 + 算 centroid，視資料量而定），代表賣家上傳後不會立刻知道「已驗證」的結果，需要一個非同步/背景工作的機制（先回應「上傳成功，驗證中」，跑完再更新狀態），不能設計成同步等待整個掃描完成才回應。

## 六、實作時需要決定的事（不算完全開放，但要做出明確選擇並記錄下來）

- **「已驗證」的判定標準**：三支偵測器都跑完之後，什麼結果算「通過」？例如：`materialized_trigger_detector` 有命中就直接判定不通過？`blind_feature_detector`/`clean_reference_detector` 的分數要用什麼門檻決定要不要卡關，還是只標記給人工複審、不卡上架流程？這幾個規則要明確定義出來，不要模稜兩可。
- **驗證報告要不要保留給使用者看**：掃描完的詳細結果（哪些樣本被標記、各偵測器的診斷數值）要不要以某種形式呈現給賣家或買家看，還是只用來決定 verified/unverified 這個二元狀態。
- **上傳流程的非同步設計**：見第五節，訓練需要時間，API/UI 要怎麼處理「上傳後還在驗證中」這個中間狀態。

## 七、技術棧延續

沿用現有 repo 的技術選擇：Python + FastAPI 後端、PyTorch 做模型訓練/推論、前端維持現有的純 HTML/CSS/JS 風格（不要引入前端框架）。`server/requirements.txt`（`fastapi`、`uvicorn`、`torch`、`torchvision`、`numpy`、`pillow`、`python-multipart`）可以沿用，缺什麼再加。
