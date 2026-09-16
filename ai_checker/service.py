"""Сервис анализа фото домашней работы через Claude API.

Изолирован от FastAPI — можно перенести в metrica-backend
и вызывать из любого роутера или фоновой задачи.
"""

import base64
import json
import logging
import os
import re
from io import BytesIO

import pillow_heif
from anthropic import AsyncAnthropic
from pydantic import ValidationError
from PIL import Image

from .prompts import (
    EGE_PROFILE_ADDENDUM,
    PROGRESS_ANALYSIS_PROMPT,
    SYSTEM_PROMPT,
    TASK_PHOTO_ADDENDUM,
    USER_PROMPT,
)
from .schemas import HomeworkReview, ProgressReport
from .taxonomy import PROGRESS_REPORT_THRESHOLD

logger = logging.getLogger(__name__)

MODES = {"school", "ege_profile"}

DEFAULT_MODEL = os.getenv("AI_CHECKER_MODEL", "claude-sonnet-5")
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # лимит Claude API на изображение

SUPPORTED_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}

# Claude API HEIC/HEIF не принимает — конвертируем в JPEG перед отправкой.
HEIC_MEDIA_TYPES = {"image/heic", "image/heif"}

pillow_heif.register_heif_opener()


class ImageTooLargeError(ValueError):
    pass


class UnsupportedImageError(ValueError):
    pass


def _convert_heic_to_jpeg(image_bytes: bytes) -> bytes:
    """Конвертирует HEIC/HEIF (формат фото на iPhone) в JPEG."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise UnsupportedImageError(f"Не удалось прочитать HEIC-файл: {e}") from e

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def prepare_image(image_bytes: bytes, media_type: str) -> tuple[bytes, str]:
    """Конвертирует HEIC при необходимости и валидирует формат/размер фото."""
    if media_type in HEIC_MEDIA_TYPES:
        image_bytes = _convert_heic_to_jpeg(image_bytes)
        media_type = "image/jpeg"

    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise UnsupportedImageError(
            f"Формат {media_type} не поддерживается. "
            "Нужен JPEG, PNG, WebP, GIF или HEIC."
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageTooLargeError(
            "Фото больше 5 МБ. Сожми изображение или сними в меньшем разрешении."
        )
    return image_bytes, media_type


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


def _image_block(image_bytes: bytes, media_type: str) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(image_bytes).decode(),
        },
    }


# Anthropic SDK отказывается делать обычный (не-streaming) запрос, если
# по формуле 3600*max_tokens/128000 ожидаемое время может превысить 10 минут
# (см. anthropic._base_client._calculate_nonstreaming_timeout) — выше этого
# порога обязателен streaming API.
STREAMING_REQUIRED_ABOVE_TOKENS = 21_333


async def _create_message(
    client: AsyncAnthropic, *, model: str, max_tokens: int, system: str, messages: list[dict], extra: dict
):
    """client.messages.create(), но через streaming API, если max_tokens
    настолько велик, что обычный запрос SDK не разрешает (см. выше)."""
    if max_tokens > STREAMING_REQUIRED_ABOVE_TOKENS:
        async with client.messages.stream(
            model=model, max_tokens=max_tokens, system=system, messages=messages, **extra
        ) as stream:
            return await stream.get_final_message()
    return await client.messages.create(
        model=model, max_tokens=max_tokens, system=system, messages=messages, **extra
    )


def _build_content(
    images: list[tuple[bytes, str]],
    task_image_bytes: bytes | None,
    task_media_type: str | None,
) -> list[dict]:
    """Собирает content-блоки запроса: если есть фото условия — сначала оно
    (с текстовой подписью), затем страницы решения ученика по порядку."""
    content: list[dict] = []
    if task_image_bytes is not None:
        content.append({"type": "text", "text": "Условие задания:"})
        content.append(_image_block(task_image_bytes, task_media_type))
    if len(images) == 1:
        content.append({"type": "text", "text": "Решение ученика:"})
        content.append(_image_block(*images[0]))
    else:
        content.append(
            {"type": "text", "text": f"Решение ученика, {len(images)} стр.:"}
        )
        for i, (image_bytes, media_type) in enumerate(images, start=1):
            content.append({"type": "text", "text": f"Страница {i}:"})
            content.append(_image_block(image_bytes, media_type))
    content.append({"type": "text", "text": USER_PROMPT})
    return content


async def review_homework(
    images: list[tuple[bytes, str]],
    *,
    mode: str = "school",
    task_image_bytes: bytes | None = None,
    task_media_type: str | None = None,
    model: str | None = None,
    client: AsyncAnthropic | None = None,
) -> HomeworkReview:
    """Анализирует фото домашки (одно или несколько, например страницы) —
    ошибки + задания для работы над ними.

    mode:
        "school"      — обычная школьная домашка;
        "ege_profile" — подготовка к профильному ЕГЭ: оценка по критериям
                        ФИПИ, проверка полноты обоснований, extended thinking.

    Если передано task_image_bytes (фото условия — на случай, когда в кадре
    решения самого условия не видно), модель сверяет решение именно с ним.
    """
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим: {mode}. Доступны: {MODES}")
    if not images:
        raise ValueError("Нужно хотя бы одно фото решения.")
    images = [prepare_image(b, mt) for b, mt in images]
    if task_image_bytes is not None:
        task_image_bytes, task_media_type = prepare_image(
            task_image_bytes, task_media_type
        )

    client = client or AsyncAnthropic()  # ключ берётся из ANTHROPIC_API_KEY

    system_prompt = SYSTEM_PROMPT
    extra: dict = {}
    max_tokens = 8000
    if mode == "ege_profile":
        system_prompt += EGE_PROFILE_ADDENDUM
        # Сложные задачи (№13-19) требуют, чтобы модель сначала сама
        # прорешала задание — включаем расширенное рассуждение.
        extra["thinking"] = {"type": "adaptive"}
        extra["output_config"] = {"effort": "high"}
        # С запасом: при effort=high расширенное рассуждение может съесть
        # большую часть бюджета токенов, не оставив места на финальный JSON
        # (наблюдали пустой ответ и "Expecting value" на max_tokens=20000).
        max_tokens = 32000
    if task_image_bytes is not None:
        system_prompt += TASK_PHOTO_ADDENDUM

    response = await _create_message(
        client,
        model=model or DEFAULT_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": _build_content(
                    images, task_image_bytes, task_media_type
                ),
            }
        ],
        extra=extra,
    )

    raw = "".join(block.text for block in response.content if block.type == "text")
    if not raw.strip():
        # Модель не успела дописать финальный JSON — обычно исчерпала
        # max_tokens на рассуждение (response.stop_reason == "max_tokens").
        raise RuntimeError(
            "Модель не завершила разбор в рамках лимита токенов "
            f"(stop_reason={response.stop_reason}). Попробуй ещё раз."
        )
    try:
        data = _extract_json(raw)
        return HomeworkReview.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        logger.error("Не удалось разобрать ответ модели. Сырой ответ:\n%s", raw)
        raise


def _format_errors_for_analysis(errors: list[dict]) -> str:
    if not errors:
        return "Ошибок не зафиксировано — ученик пока решает всё верно."
    return "\n".join(
        f"{e['homework_label']} — тема: {e['topic']} — тип: {e['severity']} — {e['explanation']}"
        for e in errors
    )


async def analyze_progress(
    student_id: str,
    homework_count: int,
    errors: list[dict],
    *,
    model: str | None = None,
    client: AsyncAnthropic | None = None,
) -> ProgressReport:
    """Группирует накопленные ошибки ученика за период в смысловые группы
    и пишет сводку прогресса. Текстовый вызов, без фото."""
    client = client or AsyncAnthropic()

    response = await client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=4000,
        system=PROGRESS_ANALYSIS_PROMPT,
        messages=[
            {"role": "user", "content": _format_errors_for_analysis(errors)}
        ],
    )

    raw = "".join(block.text for block in response.content if block.type == "text")
    try:
        data = _extract_json(raw)
        return ProgressReport.model_validate(
            {
                **data,
                "student_id": student_id,
                "period_label": f"занятия 1-{homework_count}",
                "homework_count": homework_count,
                "threshold_reached": homework_count >= PROGRESS_REPORT_THRESHOLD,
            }
        )
    except (json.JSONDecodeError, ValidationError):
        logger.error("Не удалось разобрать сводку прогресса. Сырой ответ:\n%s", raw)
        raise
