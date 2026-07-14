# AI Product Intelligence Streamlit App

Run the web UI:

```powershell
pip install -r requirements.txt
streamlit run product_intelligence_app.py
```

The app includes:

- Drag-and-drop product image upload
- Real-time category, product type, dominant color, confidence, and description
- Optional Grok/xAI vision analysis for more accurate uploaded-image descriptions
- Search with category filtering and a visual product gallery
- Catalog visualization dashboard with compression and distribution charts
- Performance metric cards for inference speed, compression accuracy, and time savings

## Data Sources

The bundled demo catalog works immediately, but it is only for UI testing. It uses a small synthetic catalog, so it cannot produce production-quality classification for arbitrary real product photos.

For the original Kaggle fashion dataset workflow, choose **Custom dataset** in the sidebar and provide:

- `styles.csv`
- an `images` folder containing product images named like `<id>.jpg`

## External API Configuration

No external API keys are required for local/demo mode.

For higher-quality uploaded-image descriptions, enable **Use Grok for uploads** and provide an xAI API key. On Streamlit Cloud, add this secret:

```toml
XAI_API_KEY = "your_xai_api_key"
```

When Grok upload analysis is enabled, the uploaded image is sent to `https://api.x.ai/v1/responses` for image understanding. Keep it disabled when you do not want uploaded images sent to an external API.

The CLIP + BLIP toggle uses local Hugging Face model weights through `torch` and `transformers`. The app intentionally checks the local cache only so Streamlit does not appear frozen while downloading large model files.

For best accuracy:

1. Use the real fashion catalog dataset, not the bundled demo catalog.
2. Cache `openai/clip-vit-base-patch32` locally for visual/text search.
3. Cache `Salesforce/blip-image-captioning-base` locally if generated descriptions are needed.
4. Increase the custom catalog indexing limit after confirming the app runs on a smaller subset.
