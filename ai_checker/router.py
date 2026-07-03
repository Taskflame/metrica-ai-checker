"""FastAPI-роутер ИИ-агента.

Для интеграции в metrica-backend достаточно:
    from ai_checker.router import router as ai_router
    app.include_router(ai_router, prefix="/api")
"""

import json
import os

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .schemas import HomeworkReview
from .service import review_homework
from .service_openai_compat import review_homework_openai_compat

router = APIRouter(tags=["ai-checker"])


@router.post("/check", response_model=HomeworkReview)
async def check_homework(
    photo: UploadFile = File(...),
    mode: str = Form("school"),  # school | ege_profile
) -> HomeworkReview:
    """Принимает фото домашки, возвращает разбор ошибок и задания."""
    image_bytes = await photo.read()
    if not image_bytes:
        raise HTTPException(400, "Пустой файл.")

    media_type = photo.content_type or "image/jpeg"

    provider = os.getenv("AI_PROVIDER", "anthropic")

    try:
        if provider == "openai_compat":
            return await review_homework_openai_compat(
                image_bytes, media_type, mode=mode
            )
        return await review_homework(image_bytes, media_type, mode=mode)
    except ValueError as e:
        # включая ImageTooLargeError и UnsupportedImageError
        raise HTTPException(400, str(e))
    except json.JSONDecodeError:
        raise HTTPException(
            502, "Модель вернула некорректный ответ. Попробуй ещё раз."
        )
    except Exception as e:  # сетевые ошибки, невалидный ключ и т.п.
        raise HTTPException(502, f"Ошибка обращения к модели ({provider}): {e}")
