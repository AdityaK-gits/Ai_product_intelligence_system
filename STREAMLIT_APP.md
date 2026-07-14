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

The bundled demo catalog works immediately.

For the original Kaggle fashion dataset workflow, choose **Custom dataset** in the sidebar and provide:

- `styles.csv`
- an `images` folder containing product images named like `<id>.jpg`

## External API Configuration

No external API keys are required.

The CLIP + BLIP toggle runs local Hugging Face models through `torch` and `transformers`. The first run may need internet access to download model weights unless they are already cached locally.
