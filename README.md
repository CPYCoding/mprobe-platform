# MProbe Platform

Front-end for the **MProbe** marketplace — a security-certified store for ML models and datasets. Anyone can list a model or dataset; every listing is scanned by MProbe (integrity + backdoor/poison detection) before it goes live.

> ⚠️ This is a **static front-end prototype**. All metrics, prices, and scan/inference results are placeholders generated in the browser — wire them to the real MProbe backend before launch.

## Run it

No build step. Open the files directly in a browser:

- `marketplace.html` — the main app: browse, filter, cart, and a seller/buyer profile (library, selling, purchase history)
- `upload.html` — list a model/dataset → simulated MProbe security scan → certified & listed
- `models/*.html` — each model's **Run (inference)** page

Because pages link to each other with relative paths, open them from this folder (e.g. double-click `marketplace.html`).

## Structure

```
platform/
├── marketplace.html      # main store + profile
├── upload.html           # "list a model" flow (upload → security scan → certify)
└── models/               # per-model Run pages (edit one page in isolation)
    ├── person-detector.html
    ├── red-car-detector.html
    ├── dataset-cleaning.html
    ├── run.css           # shared look for run pages
    └── run.js            # shared behaviour (reads each page's window.MODEL)
```

## Editing a model's Run page

Each `models/*.html` has a `window.MODEL` config block at the top — change the title, dataset, sample frames, and text there. Shared styling lives in `run.css`, shared behaviour in `run.js`.

Detector pages support **single-frame** inference and **batch dataset upload**. The data-cleaning page runs an upload → clean flow.
