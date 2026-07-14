# AI Product Intelligence Streamlit App

Run the web UI:

```powershell
pip install -r requirements.txt
streamlit run product_intelligence_app.py
```

The app includes:

- Drag-and-drop product image upload
- Real-time category, product type, dominant color, confidence, and description
- Search with category filtering and a visual product gallery
- Catalog visualization dashboard with compression and distribution charts
- Performance metric cards for inference speed, compression accuracy, and time savings

## Data Sources

The bundled demo catalog works immediately, but it is only for UI testing. It uses a small synthetic catalog, so it cannot produce production-quality classification for arbitrary real product photos.

For the original Kaggle fashion dataset workflow, choose **Custom dataset** in the sidebar and provide:

- `styles.csv`
- an `images` folder containing product images named like `<id>.jpg`

## External API Configuration

No external API keys are required.

The CLIP + BLIP toggle uses local Hugging Face model weights through `torch` and `transformers`. The app intentionally checks the local cache only so Streamlit does not appear frozen while downloading large model files.

For best accuracy:

1. Use the real fashion catalog dataset, not the bundled demo catalog.
2. Cache `openai/clip-vit-base-patch32` locally for visual/text search.
3. Cache `Salesforce/blip-image-captioning-base` locally if generated descriptions are needed.
4. Increase the custom catalog indexing limit after confirming the app runs on a smaller subset.
