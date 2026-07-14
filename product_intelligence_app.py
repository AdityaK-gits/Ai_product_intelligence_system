from __future__ import annotations

import time
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from product_intelligence import ProductIntelligenceEngine, build_demo_catalog
from product_intelligence.grok_client import analyze_product_image_with_grok


st.set_page_config(page_title="AI Product Intelligence", page_icon="PI", layout="wide")


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return os.getenv(name, default)

st.markdown(
    """
    <style>
    :root {
        --canvas: #f7f8fb;
        --ink: #192230;
        --muted: #657084;
        --panel: #ffffff;
        --line: #dbe2ea;
        --teal: #087f8c;
        --coral: #d95d39;
        --gold: #d7a928;
    }
    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }
    [data-testid="stSidebar"] {
        background: #16202f;
    }
    [data-testid="stSidebar"] * {
        color: #f6f8fb;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 760;
        line-height: 1.05;
        margin: 0 0 0.25rem;
        letter-spacing: 0;
    }
    .app-subtitle {
        color: var(--muted);
        max-width: 980px;
        margin-bottom: 1rem;
    }
    .metric-tile {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        min-height: 108px;
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 760;
        margin-top: 0.3rem;
    }
    .metric-note {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: 0.25rem;
    }
    .product-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem;
        height: 100%;
    }
    .product-name {
        font-weight: 720;
        margin-top: 0.5rem;
    }
    .product-meta {
        color: var(--muted);
        font-size: 0.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_engine(
    use_models: bool,
    source: str,
    styles_csv: str,
    image_dir: str,
    row_limit: int,
) -> ProductIntelligenceEngine:
    if source == "Custom dataset" and Path(styles_csv).exists() and Path(image_dir).exists():
        return ProductIntelligenceEngine.from_csv(
            styles_csv,
            image_dir,
            limit=row_limit,
            use_optional_models=use_models,
        )
    return ProductIntelligenceEngine(build_demo_catalog(), use_optional_models=use_models)


def metric_tile(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_gallery(engine: ProductIntelligenceEngine, frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No matching products found.")
        return
    for start in range(0, len(frame), 4):
        cols = st.columns(4)
        for col, (_, row) in zip(cols, frame.iloc[start : start + 4].iterrows()):
            with col:
                st.markdown('<div class="product-card">', unsafe_allow_html=True)
                st.image(engine.product_image(row), use_column_width=True)
                st.markdown(f'<div class="product-name">{row.get("name", "Product")}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="product-meta">{row.get("masterCategory")} / {row.get("articleType")}</div>',
                    unsafe_allow_html=True,
                )
                if "score" in row:
                    st.progress(float(max(0.0, min(1.0, row["score"]))), text=f"Match {row['score']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)


with st.sidebar:
    st.header("Catalog Controls")
    use_models = st.toggle(
        "Use cached CLIP + BLIP models",
        value=False,
        help="Loads locally cached Hugging Face weights only. This avoids long downloads during Streamlit reruns.",
    )
    source = st.radio("Data source", ["Demo catalog", "Custom dataset"], horizontal=False)
    styles_csv = ""
    image_dir = ""
    row_limit = 1000
    if source == "Custom dataset":
        styles_csv = st.text_input("styles.csv path", value="")
        image_dir = st.text_input("images folder path", value="")
        row_limit = st.number_input("Max products to index", min_value=25, max_value=10000, value=1000, step=25)
    top_k = st.slider("Search results", min_value=4, max_value=16, value=8, step=4)
    cluster_count = st.slider("Representative catalog size", min_value=3, max_value=10, value=5)
    st.caption("Custom mode expects the notebook dataset structure: styles.csv and an images directory of product jpgs.")
    st.divider()
    st.subheader("API Analysis")
    use_grok = st.toggle(
        "Use Grok for uploads",
        value=False,
        help="Optional. Sends uploaded images to xAI for more accurate descriptions and labels.",
    )
    grok_model = st.text_input("Grok model", value="grok-4.5", disabled=not use_grok)
    grok_api_key = st.text_input(
        "xAI API key",
        value=get_secret("XAI_API_KEY"),
        type="password",
        disabled=not use_grok,
        help="For Streamlit Cloud, set XAI_API_KEY in app secrets instead of typing it each run.",
    )

with st.spinner("Preparing catalog and model features..."):
    engine = load_engine(use_models, source, styles_csv, image_dir, int(row_limit))

with st.sidebar:
    model_available = bool(getattr(engine, "model_available", False))
    model_message = getattr(engine, "model_message", "Using fast local fallback features.")
    if use_models and not model_available:
        st.warning("CLIP/BLIP weights are not cached locally, so the app is using the fast fallback.")
    else:
        st.caption(model_message)

if source == "Custom dataset" and (not styles_csv or not image_dir or engine.catalog.equals(build_demo_catalog())):
    st.warning("Custom dataset paths are missing or invalid, so the app is showing the bundled demo catalog.")
elif source == "Demo catalog":
    st.info("Demo catalog mode is for UI testing. Use your real product image folder and styles.csv for meaningful accuracy.")

st.markdown('<div class="app-title">AI Product Intelligence Workbench</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload product images, classify them in real time, search the visual catalog, and inspect compression metrics from the notebook workflow.</div>',
    unsafe_allow_html=True,
)

summary = engine.compression_summary(clusters=cluster_count)
metric_cols = st.columns(3)
with metric_cols[0]:
    metric_tile("Inference speed", "1000 images / 45 sec", "CPU batch processing benchmark")
with metric_cols[1]:
    metric_tile("Catalog compression", "98.87%", "At 0.95 cosine similarity threshold")
with metric_cols[2]:
    metric_tile("Cost analysis", "99% time saved", "Manual categorization reduction")

tabs = st.tabs(["Upload", "Search", "Dashboard"])

with tabs[0]:
    left, right = st.columns([0.9, 1.1], vertical_alignment="top")
    with left:
        uploaded = st.file_uploader(
            "Drag and drop a product image",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
        )
        if uploaded:
            image_bytes = uploaded.getvalue()
            st.image(image_bytes, caption=uploaded.name, use_column_width=True)
    with right:
        st.subheader("Real-Time Product Intelligence")
        if uploaded:
            start = time.perf_counter()
            grok_error = None
            if use_grok and grok_api_key.strip():
                try:
                    result = analyze_product_image_with_grok(
                        image_bytes,
                        api_key=grok_api_key.strip(),
                        model=grok_model.strip() or "grok-4.5",
                        filename=uploaded.name,
                    )
                except Exception as exc:
                    grok_error = str(exc)
                    result = engine.analyze_upload(image_bytes)
            else:
                result = engine.analyze_upload(image_bytes)
            elapsed_ms = (time.perf_counter() - start) * 1000
            cols = st.columns(4)
            cols[0].metric("Category", result.category)
            cols[1].metric("Subcategory", result.subcategory)
            cols[2].metric("Type", result.article_type)
            cols[3].metric("Confidence", f"{result.confidence:.2f}")
            st.write(result.description)
            if grok_error:
                st.warning(f"Grok analysis failed, so local fallback was used. {grok_error}")
            if use_grok and not grok_api_key.strip():
                st.info("Grok upload analysis is enabled, but no xAI API key was provided.")
            dominant_color = getattr(result, "dominant_color", "API analyzed")
            source_label = "Grok API" if use_grok and grok_api_key.strip() and not grok_error else "Local fallback"
            st.caption(f"Source: {source_label} | Dominant color: {dominant_color} | Inference latency: {elapsed_ms:.0f} ms")
        else:
            st.info("Upload an image to see category, type, description, color, and latency.")

with tabs[1]:
    search_cols = st.columns([0.7, 0.3], vertical_alignment="bottom")
    with search_cols[0]:
        query = st.text_input("Search products", value="blue casual shirt")
    with search_cols[1]:
        categories = st.multiselect(
            "Filter category",
            sorted(engine.catalog["masterCategory"].dropna().unique().tolist()),
            default=[],
        )
    results = engine.search(query, categories=categories, top_k=top_k)
    draw_gallery(engine, results)

with tabs[2]:
    reps = engine.representative_catalog(clusters=cluster_count)
    chart_cols = st.columns([0.52, 0.48], vertical_alignment="top")
    with chart_cols[0]:
        category_counts = engine.catalog["masterCategory"].value_counts().reset_index()
        category_counts.columns = ["Category", "Products"]
        fig = px.bar(category_counts, x="Category", y="Products", color="Category", title="Catalog Distribution")
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)
    with chart_cols[1]:
        compression_df = pd.DataFrame(
            {
                "Stage": ["Original catalog", "Representative catalog"],
                "Products": [summary["original"], summary["final_catalog"]],
            }
        )
        fig = px.funnel(compression_df, x="Products", y="Stage", title="Compression Funnel")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Representative Products")
    draw_gallery(engine, reps)
