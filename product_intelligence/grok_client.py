from __future__ import annotations

import base64
import json
import mimetypes
from io import BytesIO
from urllib.parse import quote
import urllib.error
import urllib.request
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class GrokVisionResult:
    category: str
    subcategory: str
    article_type: str
    description: str
    confidence: float


def _extract_text(response_payload: dict) -> str:
    if isinstance(response_payload.get("output_text"), str):
        return response_payload["output_text"]
    chunks: list[str] = []
    for item in response_payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text") or content.get("output_text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _json_from_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _compressed_jpeg_base64(image_bytes: bytes, *, max_size: int = 1024, quality: int = 82) -> str:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((max_size, max_size))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def analyze_product_image_with_grok(
    image_bytes: bytes,
    *,
    api_key: str,
    model: str = "grok-4.5",
    filename: str = "product.jpg",
    timeout_seconds: int = 60,
) -> GrokVisionResult:
    mime_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "Analyze this uploaded image for an ecommerce product intelligence UI. "
        "If the image is not actually a product photo, say what it is instead. "
        "Return only valid JSON with keys category, subcategory, article_type, "
        "description, and confidence. Description should be one concise sentence "
        "grounded only in visible image details. Confidence must be a number from 0 to 1."
    )
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{encoded}",
                        "detail": "high",
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        "https://api.x.ai/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"xAI API request failed with HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"xAI API request failed: {exc.reason}") from exc

    text = _extract_text(payload)
    if not text:
        raise RuntimeError("xAI API returned no text output.")
    parsed = _json_from_text(text)
    return GrokVisionResult(
        category=str(parsed.get("category", "Unknown")),
        subcategory=str(parsed.get("subcategory", "Unknown")),
        article_type=str(parsed.get("article_type", "Unknown")),
        description=str(parsed.get("description", text)),
        confidence=float(parsed.get("confidence", 0.0)),
    )


def analyze_product_image_with_groq(
    image_bytes: bytes,
    *,
    api_key: str,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    filename: str = "product.jpg",
    timeout_seconds: int = 60,
) -> GrokVisionResult:
    mime_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "Analyze this uploaded image for an ecommerce product intelligence UI. "
        "If the image is not actually a product photo, say what it is instead. "
        "Return only valid JSON with keys category, subcategory, article_type, "
        "description, and confidence. Description should be one concise sentence "
        "grounded only in visible image details. Confidence must be a number from 0 to 1."
    )
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded}",
                        },
                    },
                ],
            }
        ],
        "temperature": 0.2,
        "max_completion_tokens": 600,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq API request failed with HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Groq API request failed: {exc.reason}") from exc

    text = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not text:
        raise RuntimeError("Groq API returned no text output.")
    parsed = _json_from_text(text)
    return GrokVisionResult(
        category=str(parsed.get("category", "Unknown")),
        subcategory=str(parsed.get("subcategory", "Unknown")),
        article_type=str(parsed.get("article_type", "Unknown")),
        description=str(parsed.get("description", text)),
        confidence=float(parsed.get("confidence", 0.0)),
    )


def analyze_product_image_with_gemini(
    image_bytes: bytes,
    *,
    api_key: str,
    model: str = "gemini-3.5-flash",
    filename: str = "product.jpg",
    timeout_seconds: int = 25,
) -> GrokVisionResult:
    encoded = _compressed_jpeg_base64(image_bytes)
    prompt = (
        "Analyze this uploaded image for an ecommerce product intelligence UI. "
        "If the image is not actually a product photo, say what it is instead. "
        "Return only valid JSON with keys category, subcategory, article_type, "
        "description, and confidence. Description should be one concise sentence "
        "grounded only in visible image details. Confidence must be a number from 0 to 1."
    )
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": encoded}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json",
        }
    }
    candidates = [model, "gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    models_to_try = list(dict.fromkeys(candidate for candidate in candidates if candidate))
    errors = []
    payload = None
    selected_model = None
    for candidate_model in models_to_try:
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{quote(candidate_model, safe='')}:generateContent?key={quote(api_key, safe='')}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            selected_model = candidate_model
            break
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            errors.append(f"{candidate_model}: HTTP {exc.code}: {message}")
            if exc.code not in (400, 404, 429, 500, 502, 503, 504):
                break
        except urllib.error.URLError as exc:
            errors.append(f"{candidate_model}: {exc.reason}")
            break
    if payload is None:
        raise RuntimeError("Gemini API request failed for all model candidates. " + " | ".join(errors))

    text = ""
    for candidate in payload.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if isinstance(part.get("text"), str):
                text = part["text"]
                break
        if text:
            break
    if not text:
        raise RuntimeError("Gemini API returned no text output.")
    parsed = _json_from_text(text)
    description = str(parsed.get("description", text))
    if selected_model:
        description = f"{description} (Gemini model: {selected_model})"
    return GrokVisionResult(
        category=str(parsed.get("category", "Unknown")),
        subcategory=str(parsed.get("subcategory", "Unknown")),
        article_type=str(parsed.get("article_type", "Unknown")),
        description=description,
        confidence=float(parsed.get("confidence", 0.0)),
    )
