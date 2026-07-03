"""Сервис анализа фото домашней работы через Claude API.

Изолирован от FastAPI — можно перенести в metrica-backend
и вызывать из любого роутера или фоновой задачи.
"""

import base64
import json
import os
import re

from anthropic import AsyncAnthropic

from .prompts import EGE_PROFILE_ADDENDUM, SYSTEM_PROMPT, USER_PROMPT
from .schemas import HomeworkReview

MODES = {"school", "ege_profile"}

DEFAULT_MODEL = os.getenv("AI_CHECKER_MODEL", "claude-sonnet-5")
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # лимит Claude API на изображение

SUPPORTED_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}


class ImageTooLargeError(ValueError):
    pass


class UnsupportedImageError(ValueError):
    pass


def _extract_json(text: str) -> dict:
    """Достаёт JSON из ответа модели (на случай обёртки в ```json ...```)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # если модель что-то написала вокруг JSON — берём от первой { до последней }
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


async def review_homework(
    image_bytes: bytes,
    media_type: str,
    *,
    mode: str = "school",
    model: str | None = None,
    client: AsyncAnthropic | None = None,
) -> HomeworkReview:
    """Анализирует фото домашки: ошибки + задания для работы над ними.

    mode:
        "school"      — обычная школьная домашка;
        "ege_profile" — подготовка к профильному ЕГЭ: оценка по критериям
                        ФИПИ, проверка полноты обоснований, extended thinking.
    """
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

    client = client or AsyncAnthropic()  # ключ берётся из ANTHROPIC_API_KEY

    system_prompt = SYSTEM_PROMPT
    extra: dict = {}
    max_tokens = 8000
    if mode == "ege_profile":
        system_prompt = SYSTEM_PROMPT + EGE_PROFILE_ADDENDUM
        # Сложные задачи (№13-19) требуют, чтобы модель сначала сама
        # прорешала задание — включаем расширенное рассуждение.
        extra["thinking"] = {"type": "enabled", "budget_tokens": 10000}
        max_tokens = 20000

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        **extra,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.standard_b64encode(image_bytes).decode(),
                        },
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ],
    )

    raw = "".join(block.text for block in response.content if block.type == "text")
    data = _extract_json(raw)
    return HomeworkReview.model_validate(data)
