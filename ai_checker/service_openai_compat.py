"""Бесплатный движок: любые OpenAI-совместимые API.

Работает с:
- OpenRouter (https://openrouter.ai) — бесплатные варианты Qwen-VL и др.;
- Ollama (https://ollama.com) — модель локально, без интернета и оплаты;
- любым другим OpenAI-совместимым сервером (vLLM, LM Studio, ...).

Настройка в .env:
    AI_PROVIDER=openai_compat
    OPENAI_COMPAT_BASE_URL=https://openrouter.ai/api/v1
    OPENAI_COMPAT_API_KEY=sk-or-...
    AI_CHECKER_MODEL=qwen/qwen2.5-vl-72b-instruct:free
"""

import asyncio
import base64
import os

import httpx

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

from .prompts import EGE_PROFILE_ADDENDUM, SYSTEM_PROMPT, USER_PROMPT
from .schemas import HomeworkReview
from .service import (
    MAX_IMAGE_BYTES,
    MODES,
    SUPPORTED_MEDIA_TYPES,
    ImageTooLargeError,
    UnsupportedImageError,
    _extract_json,
)


async def review_homework_openai_compat(
    image_bytes: bytes,
    media_type: str,
    *,
    mode: str = "school",
    model: str | None = None,
) -> HomeworkReview:
    """То же, что review_homework, но через OpenAI-совместимый API."""
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим: {mode}. Доступны: {MODES}")
    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise UnsupportedImageError(
            f"Формат {media_type} не поддерживается. Нужен JPEG, PNG, WebP или GIF."
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageTooLargeError(
            "Фото больше 5 МБ. Сожми изображение или сними в меньшем разрешении."
        )

    base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("OPENAI_COMPAT_API_KEY", "ollama")
    model = model or os.getenv("AI_CHECKER_MODEL", "qwen2.5vl:7b")

    system_prompt = SYSTEM_PROMPT
    if mode == "ege_profile":
        system_prompt = SYSTEM_PROMPT + EGE_PROFILE_ADDENDUM

    data_url = (
        f"data:{media_type};base64,"
        + base64.standard_b64encode(image_bytes).decode()
    )

    payload = {
        "model": model,
        "max_tokens": 8000,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
    }

    async with httpx.AsyncClient(timeout=600) as http:
        for attempt in range(1, MAX_RETRIES + 1):
            resp = await http.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)
                continue
            resp.raise_for_status()
            body = resp.json()
            break

    raw = body["choices"][0]["message"]["content"]
    data = _extract_json(raw)
    return HomeworkReview.model_validate(data)
