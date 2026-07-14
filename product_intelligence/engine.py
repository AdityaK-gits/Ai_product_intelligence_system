from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from product_intelligence.sample_data import color_name


@dataclass(frozen=True)
class AnalysisResult:
    category: str
    subcategory: str
    article_type: str
    description: str
    confidence: float
    dominant_color: str


def _open_image(payload: bytes | str | Path | Image.Image) -> Image.Image:
    if isinstance(payload, Image.Image):
        return payload.convert("RGB")
    if isinstance(payload, (str, Path)):
        return Image.open(payload).convert("RGB")
    return Image.open(BytesIO(payload)).convert("RGB")


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def visual_embedding(image: Image.Image) -> np.ndarray:
    """Fast local fallback embedding built from color, texture, and aspect features."""
    image = image.convert("RGB").resize((96, 96))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=24, range=(0.0, 1.0), density=True)
        hist_parts.append(hist.astype(np.float32))
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    vector = np.concatenate(hist_parts + [means, stds, np.array([gx, gy], dtype=np.float32)])
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class OptionalVisionModels:
    """Lazy wrapper for the notebook's CLIP and BLIP models.

    The app remains usable without these heavy dependencies. When installed and
    cached locally, users can enable model-backed captions and embeddings.
    """

    def __init__(self, *, local_files_only: bool = True) -> None:
        self.available = False
        self.caption_available = False
        self.error: str | None = None
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                local_files_only=local_files_only,
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                local_files_only=local_files_only,
            )
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on optional local model cache
            self.error = f"CLIP unavailable: {exc}"

        if not self.available:
            return

        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            self.blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                local_files_only=local_files_only,
            )
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                local_files_only=local_files_only,
            ).to(self.device)
            self.caption_available = True
        except Exception as exc:  # pragma: no cover - depends on optional local model cache
            self.error = f"{self.error or ''} BLIP captions unavailable: {exc}".strip()

    def image_embedding(self, image: Image.Image) -> np.ndarray:
        if not self.available:
            return visual_embedding(image)
        inputs = self.clip_processor(images=[image.convert("RGB")], return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        vector = features.cpu().numpy()[0].astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def caption(self, image: Image.Image) -> str | None:
        if not self.caption_available:
            return None
        inputs = self.blip_processor(image.convert("RGB"), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.blip_model.generate(**inputs, max_new_tokens=32)
        return self.blip_processor.decode(output[0], skip_special_tokens=True)

    def text_embedding(self, text: str) -> np.ndarray | None:
        if not self.available:
            return None
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
        vector = features.cpu().numpy()[0].astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def zero_shot_scores(self, image: Image.Image, labels: list[str]) -> np.ndarray | None:
        if not self.available:
            return None
        prompts = [f"a product photo of {label}" for label in labels]
        inputs = self.clip_processor(
            text=prompts,
            images=[image.convert("RGB")],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]


class ProductIntelligenceEngine:
    def __init__(self, catalog: pd.DataFrame, use_optional_models: bool = False) -> None:
        self.catalog = catalog.copy().reset_index(drop=True)
        self.models = OptionalVisionModels() if use_optional_models else None
        self._prepare_catalog()

    @property
    def model_available(self) -> bool:
        return bool(self.models and self.models.available)

    @property
    def model_message(self) -> str:
        if not self.models:
            return "Using fast local fallback features."
        if self.models.available:
            caption_state = "BLIP captions enabled" if self.models.caption_available else "BLIP captions not cached"
            return f"CLIP enabled on {self.models.device}; {caption_state}."
        return self.models.error or "Optional vision models are unavailable."

    @classmethod
    def from_csv(
        cls,
        styles_csv: str | Path,
        image_dir: str | Path,
        *,
        limit: int | None = 1500,
        use_optional_models: bool = False,
    ) -> "ProductIntelligenceEngine":
        styles = pd.read_csv(styles_csv, on_bad_lines="skip")
        styles["product_id"] = styles["id"].astype(str)
        rows = []
        for _, row in styles.iterrows():
            path = Path(image_dir) / f"{row['product_id']}.jpg"
            if not path.exists():
                continue
            rows.append(
                {
                    "product_id": row["product_id"],
                    "name": row.get("productDisplayName", row.get("articleType", "Product")),
                    "masterCategory": row.get("masterCategory", "Unknown"),
                    "subCategory": row.get("subCategory", "Unknown"),
                    "articleType": row.get("articleType", "Unknown"),
                    "tags": " ".join(str(row.get(col, "")) for col in ("gender", "baseColour", "season", "usage")),
                    "image_path": str(path),
                }
            )
            if limit and len(rows) >= limit:
                break
        return cls(pd.DataFrame(rows), use_optional_models=use_optional_models)

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        if self.models:
            return self.models.image_embedding(image)
        return visual_embedding(image)

    def _prepare_catalog(self) -> None:
        if self.catalog.empty:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.text_matrix = None
            self.vectorizer = None
            return
        embeddings = []
        texts = []
        for _, row in self.catalog.iterrows():
            image = self.product_image(row)
            embeddings.append(self._embed_image(image))
            texts.append(self._row_text(row))
        self.embeddings = np.vstack(embeddings).astype(np.float32)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.text_matrix = self.vectorizer.fit_transform(texts)

    def _row_text(self, row: pd.Series) -> str:
        fields = ["name", "masterCategory", "subCategory", "articleType", "tags"]
        return " ".join(str(row.get(field, "")) for field in fields)

    def product_image(self, row: pd.Series) -> Image.Image:
        if row.get("image_bytes") is not None:
            return _open_image(row["image_bytes"])
        return _open_image(row["image_path"])

    def analyze_upload(self, payload: bytes) -> AnalysisResult:
        image = _open_image(payload)
        embedding = self._embed_image(image)
        if len(self.catalog) == 0:
            return self._fallback_analysis(image)
        similarities = cosine_similarity([embedding], self.embeddings)[0]
        best_idx = int(np.argmax(similarities))
        best = self.catalog.iloc[best_idx]
        dominant = color_name(tuple((np.asarray(image.resize((1, 1))).reshape(3)).tolist()))
        category = str(best.get("masterCategory", "Unknown"))
        subcategory = str(best.get("subCategory", "Unknown"))
        article_type = str(best.get("articleType", "Unknown"))
        if self.model_available:
            category_labels = sorted(self.catalog["masterCategory"].dropna().astype(str).unique().tolist())
            article_labels = sorted(self.catalog["articleType"].dropna().astype(str).unique().tolist())
            category_scores = self.models.zero_shot_scores(image, category_labels) if category_labels else None
            article_scores = self.models.zero_shot_scores(image, article_labels) if article_labels else None
            if category_scores is not None:
                category = category_labels[int(np.argmax(category_scores))]
            if article_scores is not None:
                article_type = article_labels[int(np.argmax(article_scores))]
                match = self.catalog[self.catalog["articleType"].astype(str) == article_type]
                if not match.empty:
                    subcategory = str(match.iloc[0].get("subCategory", subcategory))
        caption = self.models.caption(image) if self.models else None
        description = caption or f"{dominant.title()} product visually closest to {best.get('name', best.get('articleType', 'catalog item'))}."
        return AnalysisResult(
            category=category,
            subcategory=subcategory,
            article_type=article_type,
            description=description,
            confidence=float(np.clip(similarities[best_idx], 0.0, 1.0)),
            dominant_color=dominant,
        )

    def _fallback_analysis(self, image: Image.Image) -> AnalysisResult:
        dominant = color_name(tuple((np.asarray(image.resize((1, 1))).reshape(3)).tolist()))
        return AnalysisResult("Unknown", "Unknown", "Product", f"{dominant.title()} uploaded product image.", 0.0, dominant)

    def search(self, query: str, *, categories: Iterable[str] | None = None, top_k: int = 12) -> pd.DataFrame:
        if not query.strip() or self.catalog.empty:
            return self.catalog.head(top_k).copy()
        text_scores = cosine_similarity(self.vectorizer.transform([query]), self.text_matrix)[0]
        if self.model_available:
            query_embedding = self.models.text_embedding(query)
            if query_embedding is not None and query_embedding.shape[0] == self.embeddings.shape[1]:
                clip_scores = cosine_similarity([query_embedding], self.embeddings)[0]
                text_scores = (0.35 * text_scores) + (0.65 * clip_scores)
        query_tokens = set(query.lower().split())
        color_bonus = np.array([0.08 if any(token in self._row_text(row).lower() for token in query_tokens) else 0.0 for _, row in self.catalog.iterrows()])
        scores = text_scores + color_bonus
        frame = self.catalog.copy()
        frame["score"] = scores
        if categories:
            frame = frame[frame["masterCategory"].isin(categories)]
        return frame.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

    def gallery_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in frame.iterrows():
            item = row.to_dict()
            item["image"] = self.product_image(row)
            rows.append(item)
        return pd.DataFrame(rows)

    def compression_summary(self, threshold: float = 0.95, clusters: int = 5) -> dict[str, float | int]:
        original = len(self.catalog)
        if original == 0:
            return {"original": 0, "final_catalog": 0, "reduction_pct": 0.0, "clusters": 0}
        final_catalog = max(1, min(clusters, original))
        reduction = ((original - final_catalog) / original) * 100
        return {
            "original": original,
            "final_catalog": final_catalog,
            "reduction_pct": reduction,
            "clusters": final_catalog,
            "cosine_threshold": threshold,
        }

    def representative_catalog(self, clusters: int = 5) -> pd.DataFrame:
        if self.catalog.empty:
            return self.catalog.copy()
        n_clusters = max(1, min(clusters, len(self.catalog)))
        if n_clusters == len(self.catalog):
            reps = self.catalog.copy()
            reps["cluster_id"] = range(len(reps))
            return reps
        n_components = max(1, min(8, self.embeddings.shape[1], len(self.catalog) - 1))
        reduced = PCA(n_components=n_components, random_state=42).fit_transform(self.embeddings)
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(reduced)
        rows = []
        for cluster_id in range(n_clusters):
            members = np.where(labels == cluster_id)[0]
            centroid = reduced[members].mean(axis=0)
            best_local = members[np.argmin(np.linalg.norm(reduced[members] - centroid, axis=1))]
            row = self.catalog.iloc[int(best_local)].copy()
            row["cluster_id"] = cluster_id
            row["cluster_size"] = len(members)
            rows.append(row)
        return pd.DataFrame(rows).reset_index(drop=True)
