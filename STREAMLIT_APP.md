# AI Product Intelligence Streamlit App

Run the web UI:

```powershell
pip install -r requirements.txt
streamlit run product_intelligence_app.py
```

The app includes:

- Drag-and-drop product image upload
- Real-time category, product type, dominant color, confidence, and description
- Optional Gemini, GroqCloud, or xAI Grok vision analysis for more accurate uploaded-image descriptions
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

For higher-quality uploaded-image descriptions, enable **Use API for uploads** and choose a provider.

For a Google AI Studio Gemini key, add this Streamlit Cloud secret:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
```

Use the default Gemini model:

```text
gemini-3.5-flash
```

If that model is not available to your key, the app also tries `gemini-2.5-flash` and `gemini-2.5-flash-lite`.

For a GroqCloud key that starts with `gsk_`, add this Streamlit Cloud secret:

```toml
GROQ_API_KEY = "your_groqcloud_api_key"
```

Use the default GroqCloud model:

```text
meta-llama/llama-4-scout-17b-16e-instruct
```

For an xAI/Grok key from `console.x.ai`, add this secret instead:

```toml
XAI_API_KEY = "your_xai_api_key"
```

When API upload analysis is enabled, the uploaded image is sent to the selected external provider. Keep it disabled when you do not want uploaded images sent to an external API.

The CLIP + BLIP toggle uses local Hugging Face model weights through `torch` and `transformers`. The app intentionally checks the local cache only so Streamlit does not appear frozen while downloading large model files.

For best accuracy:

1. Use the real fashion catalog dataset, not the bundled demo catalog.
2. Cache `openai/clip-vit-base-patch32` locally for visual/text search.
3. Cache `Salesforce/blip-image-captioning-base` locally if generated descriptions are needed.
4. Increase the custom catalog indexing limit after confirming the app runs on a smaller subset.
