from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class DemoProduct:
    product_id: str
    name: str
    category: str
    subcategory: str
    article_type: str
    tags: tuple[str, ...]
    color: tuple[int, int, int]


DEMO_PRODUCTS: tuple[DemoProduct, ...] = (
    DemoProduct("1001", "Blue Oxford Shirt", "Apparel", "Topwear", "Shirt", ("blue", "casual", "cotton", "office"), (55, 112, 183)),
    DemoProduct("1002", "Red Evening Dress", "Apparel", "Dress", "Dress", ("red", "party", "formal", "women"), (190, 49, 68)),
    DemoProduct("1003", "Black Running Shoes", "Footwear", "Shoes", "Sports Shoes", ("black", "sports", "running", "mesh"), (35, 37, 44)),
    DemoProduct("1004", "Tan Leather Handbag", "Accessories", "Bags", "Handbag", ("tan", "leather", "handbag", "classic"), (185, 131, 76)),
    DemoProduct("1005", "Green Training Tee", "Apparel", "Topwear", "T-Shirt", ("green", "gym", "training", "activewear"), (56, 145, 105)),
    DemoProduct("1006", "White Low Sneakers", "Footwear", "Shoes", "Casual Shoes", ("white", "sneaker", "casual", "streetwear"), (226, 226, 215)),
    DemoProduct("1007", "Navy Denim Jacket", "Apparel", "Outerwear", "Jacket", ("navy", "denim", "jacket", "layering"), (42, 66, 105)),
    DemoProduct("1008", "Gold Analog Watch", "Accessories", "Watches", "Watch", ("gold", "watch", "metal", "formal"), (212, 171, 74)),
    DemoProduct("1009", "Pink Summer Kurta", "Apparel", "Ethnic", "Kurta", ("pink", "summer", "ethnic", "cotton"), (222, 123, 158)),
    DemoProduct("1010", "Grey Laptop Backpack", "Accessories", "Bags", "Backpack", ("grey", "backpack", "travel", "utility"), (116, 124, 133)),
    DemoProduct("1011", "Orange Track Pants", "Apparel", "Bottomwear", "Track Pants", ("orange", "sports", "training", "comfort"), (224, 113, 55)),
    DemoProduct("1012", "Brown Formal Belt", "Accessories", "Belts", "Belt", ("brown", "formal", "leather", "office"), (116, 75, 45)),
)


def _product_image(product: DemoProduct, size: int = 224) -> bytes:
    image = Image.new("RGB", (size, size), product.color)
    draw = ImageDraw.Draw(image)
    accent = tuple(max(0, min(255, c + 38)) for c in product.color)
    shadow = tuple(max(0, c - 52) for c in product.color)
    draw.rounded_rectangle((22, 30, size - 22, size - 34), radius=24, fill=accent, outline=shadow, width=4)
    draw.ellipse((64, 70, 160, 166), fill=product.color, outline=shadow, width=5)
    draw.rectangle((42, 176, 182, 190), fill=shadow)
    draw.text((28, 198), product.article_type[:18], fill=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_demo_catalog() -> pd.DataFrame:
    """Return a small, image-backed catalog so the Streamlit app works first run."""
    rows = []
    for product in DEMO_PRODUCTS:
        rows.append(
            {
                "product_id": product.product_id,
                "name": product.name,
                "masterCategory": product.category,
                "subCategory": product.subcategory,
                "articleType": product.article_type,
                "tags": ", ".join(product.tags),
                "image_bytes": _product_image(product),
            }
        )
    return pd.DataFrame(rows)


def color_name(rgb: tuple[float, float, float]) -> str:
    colors = {
        "black": np.array([35, 35, 35]),
        "white": np.array([225, 225, 220]),
        "blue": np.array([55, 105, 180]),
        "red": np.array([190, 50, 65]),
        "green": np.array([55, 145, 105]),
        "tan": np.array([180, 130, 80]),
        "gold": np.array([210, 170, 75]),
        "pink": np.array([220, 120, 160]),
        "grey": np.array([120, 125, 135]),
        "orange": np.array([225, 115, 55]),
        "brown": np.array([115, 75, 45]),
        "navy": np.array([40, 65, 105]),
    }
    target = np.array(rgb)
    return min(colors, key=lambda name: float(np.linalg.norm(colors[name] - target)))
