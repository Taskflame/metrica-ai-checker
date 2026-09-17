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
import json
import logging
import os

import httpx
from pydantic import ValidationError

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

from .prompts import (
    EGE_PROFILE_ADDENDUM,
    PROGRESS_ANALYSIS_PROMPT,
    SYSTEM_PROMPT,
    TASK_PHOTO_ADDENDUM,
    USER_PROMPT,
)
from .schemas import HomeworkReview, ProgressReport
from .service import MODES, _extract_json, _format_errors_for_analysis, prepare_image
from .taxonomy import PROGRESS_REPORT_THRESHOLD

logger = logging.getLogger(__name__)


def _data_url(image_bytes: bytes, media_type: str) -> str:
    return f"data:{media_type};base64," + base64.standard_b64encode(image_bytes).decode()


def _log_usage(model: str, body: dict) -> None:
    usage = body.get("usage")
    if not usage:
        return
    logger.info(
        "Модель %s: prompt_tokens=%s, completion_tokens=%s, total_tokens=%s, cost=$%s",
        model,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
        usage.get("cost", "?"),
    )


async def _post_chat_completions(base_url: str, api_key: str, payload: dict) -> dict:
    proxy = os.getenv("OUTBOUND_PROXY_URL") or None
    async with httpx.AsyncClient(timeout=600, proxy=proxy) as http:
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
            return resp.json()
    raise RuntimeError("unreachable")  # цикл всегда либо return, либо raise_for_status


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
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _data_url(task_image_bytes, task_media_type)},
            }
        )
    if len(images) == 1:
        content.append({"type": "text", "text": "Решение ученика:"})
        content.append(
            {"type": "image_url", "image_url": {"url": _data_url(*images[0])}}
        )
    else:
        content.append(
            {"type": "text", "text": f"Решение ученика, {len(images)} стр.:"}
        )
        for i, (image_bytes, media_type) in enumerate(images, start=1):
            content.append({"type": "text", "text": f"Страница {i}:"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(image_bytes, media_type)},
                }
            )
    content.append({"type": "text", "text": USER_PROMPT})
    return content


async def review_homework_openai_compat(
    images: list[tuple[bytes, str]],
    *,
    mode: str = "school",
    task_image_bytes: bytes | None = None,
    task_media_type: str | None = None,
    model: str | None = None,
) -> HomeworkReview:
    """То же, что review_homework, но через OpenAI-совместимый API.
    images — одно или несколько фото решения (например, страницы)."""
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим: {mode}. Доступны: {MODES}")
    if not images:
        raise ValueError("Нужно хотя бы одно фото решения.")
    images = [prepare_image(b, mt) for b, mt in images]
    if task_image_bytes is not None:
        task_image_bytes, task_media_type = prepare_image(
            task_image_bytes, task_media_type
        )

    base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("OPENAI_COMPAT_API_KEY", "ollama")
    model = model or os.getenv("AI_CHECKER_MODEL", "qwen2.5vl:7b")

    system_prompt = SYSTEM_PROMPT
    if mode == "ege_profile":
        system_prompt += EGE_PROFILE_ADDENDUM
    if task_image_bytes is not None:
        system_prompt += TASK_PHOTO_ADDENDUM

    content = _build_content(images, task_image_bytes, task_media_type)

    payload = {
        "model": model,
        "max_tokens": 8000,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    }

    body = await _post_chat_completions(base_url, api_key, payload)
    _log_usage(model, body)
    raw = body["choices"][0]["message"]["content"]
    try:
        data = _extract_json(raw)
        return HomeworkReview.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        logger.error("Не удалось разобрать ответ модели (%s). Сырой ответ:\n%s", model, raw)
        raise


async def analyze_progress_openai_compat(
    student_id: str,
    homework_count: int,
    errors: list[dict],
    *,
    model: str | None = None,
) -> ProgressReport:
    """То же, что analyze_progress, но через OpenAI-совместимый API."""
    base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("OPENAI_COMPAT_API_KEY", "ollama")
    model = model or os.getenv("AI_CHECKER_MODEL", "qwen2.5vl:7b")

    payload = {
        "model": model,
        "max_tokens": 4000,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": PROGRESS_ANALYSIS_PROMPT},
            {"role": "user", "content": _format_errors_for_analysis(errors)},
        ],
    }

    body = await _post_chat_completions(base_url, api_key, payload)
    _log_usage(model, body)
    raw = body["choices"][0]["message"]["content"]
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
        logger.error("Не удалось разобрать сводку прогресса (%s). Сырой ответ:\n%s", model, raw)
        raise
