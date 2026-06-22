<div align="center">

# 🛍️ AI Product Intelligence System

### Multimodal AI for E-Commerce Product Management

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface)](https://huggingface.co)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Gen AI Bootcamp – Challenge**
> Submitted by **Aditya Kolluru** · B.Tech ISE · M.S. Ramaiah University of Applied Sciences

</div>

---

## 📌 Overview

E-commerce platforms like Amazon and Flipkart manage **millions of products** uploaded by different sellers. Manually categorizing products, writing descriptions, removing duplicates, and enabling search is expensive and slow.

This project builds an **AI-powered Product Intelligence System** that automates all of this using state-of-the-art multimodal models:

| Task | What it does |
|------|-------------|
| 🔍 **Product Understanding** | Identifies category + generates description from any product image |
| 🗂️ **Unique Catalog Creation** | Reduces 44,441 products to 500 representative items (98.87% compression) |
| 🔎 **Reverse Product Search** | Find products using natural language — *"blue casual shirt"* → top matching images |

---

## 🗃️ Dataset

**Fashion Product Images (Small)** — [Kaggle Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

| Attribute | Value |
|-----------|-------|
| Total Images | **44,441** |
| Metadata Columns | gender, masterCategory, subCategory, articleType, baseColour, season, productDisplayName |
| Format | `.jpg` images + `styles.csv` metadata |

<br>

![Dataset Preview](assets/ss01.png)
*Total images confirmed: 44,441*

<br>

![styles.head()](assets/ss07.png)
*Dataset metadata preview — styles.csv with product attributes*

---

## 🧰 Technologies Used

### AI Models

| Model | Purpose |
|-------|---------|
| **CLIP** (OpenAI `clip-vit-base-patch32`) | Image embeddings · Text embeddings · Cross-modal similarity search |
| **BLIP** (Salesforce `blip-image-captioning-base`) | Automatic product description generation from images |

### Libraries

```
PyTorch · Transformers (HuggingFace) · NumPy · Pandas
Matplotlib · Scikit-Learn · FAISS* · tqdm · Pillow
```
> *FAISS is planned for future large-scale deployment (nearest-neighbor at millions of SKUs). Current implementation uses NumPy cosine similarity.*

### ML Techniques
- **CLIP Embeddings** — 512-dimensional visual & textual feature extraction
- **PCA** — Dimensionality reduction: 512 → 128
- **K-Means Clustering** — k=500 clusters for catalog deduplication
- **Cosine Similarity** — Text-to-image matching for reverse search
- **Euclidean Distance** — Centroid-nearest product selection

<br>

![Hardware](assets/ss02.png)
*Kaggle environment — CUDA Available: True · GPU: Tesla T4*

<br>

![CLIP Loaded](assets/ss08.png)
*CLIP model and processor loaded successfully*

---

## 🏗️ System Architecture

```
Product Image
      │
      ▼
Image Preprocessing (PIL → RGB → Batch of 64)
      │
      ▼
CLIP Embedding Generation  ──────────────────────────────────┐
      │                                                       │
      ▼                                                       ▼
 [Task 1]                                              [Task 3]
Product Understanding                            Reverse Product Search
  ├── Category Retrieval (styles.csv)        Text Query → CLIP Text Embedding
  └── BLIP Description Generation            → Cosine Similarity → Top-k Products
      │
      ▼
 [Task 2]
Unique Catalog Creation
  ├── PCA: 512 → 128 dims
  ├── K-Means: 500 clusters
  └── Representative Selection (centroid-nearest)
      │
      ▼
  Final Outputs
(Product Metadata · Compact Catalog · Search Results)
```

---

## ✅ Task 1 — Product Understanding

### Objective
Automatically identify product category and generate a human-readable description from a raw product image — no manual input required.

### How it works
1. Product image loaded via `PIL`, processed through `CLIPProcessor`
2. Image filename (ID) matched against `styles.csv` → retrieves `masterCategory`, `subCategory`, `articleType`
3. `BlipForConditionalGeneration` generates a natural language caption for the image

### Result

```
IMAGE:   31973.jpg

CATEGORY
{'masterCategory': 'Footwear', 'subCategory': 'Shoes', 'articleType': 'Sports Shoes'}

DESCRIPTION
nike air pegasus black / blue
```

<br>

![Categories Loaded](assets/ss13.png)
*Category lookup built — 44,424 entries loaded from styles.csv*

<br>

![BLIP Caption](assets/ss12.png)
*BLIP-generated description: "nike air pegasus black / blue"*

<br>

![Full Output](assets/ss14.png)
*Complete product understanding output — IMAGE + CATEGORY + DESCRIPTION*

---

## ✅ Task 2 — Unique Product Catalog Creation

### Objective
Compress 44,441 products into a **500-item catalog** of representative, visually-distinct products — eliminating near-duplicate listings from multiple sellers.

### Pipeline

**Step 1 — Embedding Generation**

```python
# Batch CLIP inference over all 44,441 images
for batch in tqdm(batches):
    image_features = clip_model.get_image_features(**inputs)
    embeddings.append(image_features.cpu().numpy())

embeddings = np.vstack(embeddings)
# Result: (44441, 512)
```

<br>

![Embedding Shape](assets/ss03.png)
*Embedding generation complete — Shape: (44441, 512) · 695/695 batches in 8m 15s*

<br>

**Step 2 — PCA Dimensionality Reduction**

```python
pca = PCA(n_components=128, random_state=42)
embeddings_reduced = pca.fit_transform(embeddings)
# (44441, 512) → (44441, 128)
```

**Step 3 — K-Means Clustering**

```python
kmeans = KMeans(n_clusters=500, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings_reduced)
# Clusters Created: 500
```

**Step 4 — Representative Product Selection**

```python
for cluster_id in range(NUM_CLUSTERS):
    cluster_members = np.where(cluster_labels == cluster_id)[0]
    centroid = kmeans.cluster_centers_[cluster_id]
    distances = euclidean_distances(embeddings_reduced[cluster_members], centroid)
    representative_indices.append(cluster_members[np.argmin(distances)])
```

### 📊 Results

| Metric | Value |
|--------|-------|
| Original Products | 44,441 |
| Final Catalog Size | **500** |
| **Catalog Reduction** | **98.87%** |

<br>

![Reduction Metrics](assets/ss15.png)
*Catalog reduction confirmed — Original: 44441 → Final: 500 → Reduction: 98.87%*

<br>

![Catalog DataFrame](assets/ss05.png)
*catalog_df.head() — cluster_id mapped to representative image filenames*

<br>

![Catalog Close-up](assets/ss10.png)
*First 5 representative products: 7537.jpg, 5831.jpg, 38421.jpg, 28637.jpg, 17009.jpg*

<br>

![Product Grid](assets/ss04.png)
*20 sample representative products — diverse across clothing, footwear, and accessories*

<br>

![Product Grid Alt](assets/ss09.png)
*Catalog grid (alternate view) — pants, shirts, shoes, handbags, watches, dresses*

---

## ✅ Task 3 — Reverse Product Search

### Objective
Let users find products using **plain English text** — no image upload needed.

### How it works

```python
def search_products(query, top_k=5):
    # Encode text query with CLIP
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
    text_features = clip_model.get_text_features(**inputs).cpu().numpy()

    # Cosine similarity against all 44,441 image embeddings
    similarities = cosine_similarity(text_features, embeddings)[0]

    # Return top-k most similar product indices
    return similarities.argsort()[-top_k:][::-1]
```

> This works because CLIP was trained on 400M image-text pairs, creating a **shared embedding space** where "blue casual shirt" (text) maps close to actual blue shirt images.

### 🔍 Search Results

**Query: "blue casual shirt"**

<br>

![Search Results](assets/ss11.png)
*Top 5 results for "blue casual shirt" — blue/denim shirts retrieved with high accuracy*

<br>

**Multi-Query Benchmark**

| Query | Top-5 Indices | Observation |
|-------|--------------|-------------|
| `"blue casual shirt"` | 21704, 9352, 13542, 22004, 7006 | Blue/denim shirts |
| `"red dress"` | 18129, 23136, 25333, 26485, 29481 | Red dresses |
| `"sports shoes"` | 34716, 11574, 2035, 29554, 40182 | Athletic footwear |
| `"black handbag"` | 42049, 5168, 26272, 14104, 21789 | Dark-colored bags |

<br>

![Multi Query](assets/ss16.png)
*Multi-query benchmark — 4 queries tested, all returning highly relevant products*

---

## 📈 Evaluation Summary

### Task 1 — Product Understanding

| Metric | Result |
|--------|--------|
| Category Retrieval | ✅ 44,424 / 44,441 products matched |
| Description Generation | ✅ BLIP produced accurate captions |
| Pipeline Output | ✅ IMAGE + CATEGORY + DESCRIPTION all produced |

### Task 2 — Catalog Creation

| Metric | Value |
|--------|-------|
| Embedding Shape (raw) | (44441, 512) |
| Embedding Shape (PCA) | (44441, 128) |
| Clusters Formed | 500 |
| Representative Products | 500 |
| Catalog Reduction | **98.87%** |
| Processing Time | 8 min 15 sec (Tesla T4 GPU) |

### Task 3 — Reverse Search

| Query | Relevance |
|-------|-----------|
| "blue casual shirt" | ✅ High — blue/light shirts retrieved |
| "red dress" | ✅ High — red dresses retrieved |
| "sports shoes" | ✅ High — athletic footwear retrieved |
| "black handbag" | ✅ High — dark bags retrieved |

---

## 💼 Business Value

| Benefit | Impact |
|---------|--------|
| 🤖 **Automated Listing** | No manual category tagging or description writing for sellers |
| 🔍 **Smarter Search** | Natural language queries increase user engagement and conversion |
| 🗑️ **Duplicate Removal** | 98.87% catalog compression eliminates redundant seller listings |
| 📦 **Better Discovery** | Semantic retrieval powers recommendations beyond keyword matching |
| ⚡ **Scalability** | GPU batch processing handles tens of thousands of products efficiently |
| 💰 **Cost Savings** | Replaces large manual cataloging teams with a single AI pipeline |

---

## 🚀 Future Enhancements

- [ ] **FAISS integration** — Sub-second ANN search at millions of products
- [ ] **Fine-tune CLIP** on domain-specific fashion data for higher retrieval precision
- [ ] **Multilingual search** using multilingual CLIP variants
- [ ] **Personalized search** incorporating user click behaviour and purchase history
- [ ] **Real-time catalog updates** with incremental embedding indexing
- [ ] **A/B testing** framework to measure CTR and conversion rate improvements

---

## 📁 Project Structure

```
ai-product-intelligence/
│
├── Aditya_Kolluru_AI_Product_Intelligence.ipynb   # Main notebook (Kaggle)
│
├── assets/                    # Screenshots and output images
│   ├── ss01.png               # Dataset loaded (44,441 images)
│   ├── ss03.png               # Embedding shape output
│   ├── ss04.png               # Representative product grid
│   ├── ss11.png               # Search results visualization
│   └── ...
│
├── unique_catalog.csv         # 500-item representative catalog (output)
├── image_embeddings.npy       # Saved CLIP embeddings (output)
└── README.md
```

---

## ⚙️ Setup & Usage

```bash
# Install dependencies
pip install torch transformers scikit-learn numpy pandas matplotlib pillow tqdm faiss-cpu

# Run on Kaggle (recommended — free Tesla T4 GPU)
# Upload notebook to Kaggle, attach dataset:
# https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
```

### Key notebook cells

```python
# 1. Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Generate embeddings
embeddings = np.vstack(embeddings)   # Shape: (44441, 512)

# 3. Build catalog
kmeans = KMeans(n_clusters=500, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings_reduced)

# 4. Search products
results = search_products("blue casual shirt", top_k=5)
```

---

## 👤 Author

**Aditya Kolluru**
B.Tech Information Science & Engineering
M.S. Ramaiah University of Applied Sciences, Bengaluru

[![GitHub](https://img.shields.io/badge/GitHub-AdityaK--gits-181717?style=flat&logo=github)](https://github.com/AdityaK-gits)
[![Portfolio](https://img.shields.io/badge/Portfolio-aditya--kolluru-blue?style=flat&logo=netlify)](https://aditya-kolluru-portfolio.netlify.app)

---

<div align="center">

*Built for Gen AI Bootcamp · Day 2 Homework Challenge*

</div>
