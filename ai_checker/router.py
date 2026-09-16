"""FastAPI-роутер ИИ-агента.

Для интеграции в metrica-backend достаточно:
    from ai_checker.router import router as ai_router
    app.include_router(ai_router, prefix="/api")
"""

import json
import os

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from . import storage
from .schemas import HomeworkReview, ProgressReport
from .service import analyze_progress, review_homework
from .service_openai_compat import (
    analyze_progress_openai_compat,
    review_homework_openai_compat,
)

router = APIRouter(tags=["ai-checker"])


MAX_PHOTOS = 10


@router.post("/check", response_model=HomeworkReview)
async def check_homework(
    photos: list[UploadFile] = File(...),
    task_photo: UploadFile | None = File(None),
    mode: str = Form("school"),  # school | ege_profile
    student_id: str = Form(...),
) -> HomeworkReview:
    """Принимает одно или несколько фото домашки (например, страницы одной
    работы) и опционально фото условия, возвращает разбор ошибок и задания.
    Сохраняет результат за учеником (student_id) — без этого не получится
    копить ошибки для сводки прогресса."""
    student_id = student_id.strip()
    if not student_id:
        raise HTTPException(400, "Не указан ученик.")

    if not photos:
        raise HTTPException(400, "Нужно хотя бы одно фото.")
    if len(photos) > MAX_PHOTOS:
        raise HTTPException(400, f"Слишком много фото за раз (максимум {MAX_PHOTOS}).")

    images: list[tuple[bytes, str]] = []
    for photo in photos:
        image_bytes = await photo.read()
        if not image_bytes:
            raise HTTPException(400, "Пустой файл.")
        images.append((image_bytes, photo.content_type or "image/jpeg"))

    task_image_bytes: bytes | None = None
    task_media_type: str | None = None
    if task_photo is not None:
        task_image_bytes = await task_photo.read()
        if not task_image_bytes:
            raise HTTPException(400, "Пустой файл условия.")
        task_media_type = task_photo.content_type or "image/jpeg"

    provider = os.getenv("AI_PROVIDER", "anthropic")

    try:
        if provider == "openai_compat":
            review = await review_homework_openai_compat(
                images,
                mode=mode,
                task_image_bytes=task_image_bytes,
                task_media_type=task_media_type,
            )
        else:
            review = await review_homework(
                images,
                mode=mode,
                task_image_bytes=task_image_bytes,
                task_media_type=task_media_type,
            )
    except json.JSONDecodeError:
        # JSONDecodeError — подкласс ValueError, поэтому должен ловиться
        # раньше него, иначе наружу утечёт сырой парсер-эксепшен.
        raise HTTPException(
            502, "Модель вернула некорректный ответ. Попробуй ещё раз."
        )
    except ValueError as e:
        # включая ImageTooLargeError и UnsupportedImageError
        raise HTTPException(400, str(e))
    except Exception as e:  # сетевые ошибки, невалидный ключ и т.п.
        raise HTTPException(502, f"Ошибка обращения к модели ({provider}): {e}")

    if review.readable:
        homework_number, homework_label = await storage.save_submission(
            student_id, mode, review
        )
        review.homework_number = homework_number
        review.homework_label = homework_label

    return review


@router.get("/students/{student_id}/homeworks")
async def list_homeworks(student_id: str) -> list[dict]:
    """История проверенных домашек ученика — для UI и проверки, набралось
    ли достаточно занятий для сводки прогресса."""
    return await storage.list_submissions(student_id)


@router.get("/students/{student_id}/progress", response_model=ProgressReport)
async def get_progress(student_id: str) -> ProgressReport:
    """Группирует накопленные ошибки ученика в смысловые темы и возвращает
    сводку прогресса. Доступна и раньше порога PROGRESS_REPORT_THRESHOLD —
    сводка просто помечается threshold_reached=false."""
    submissions = await storage.list_submissions(student_id)
    if not submissions:
        raise HTTPException(404, "У этого ученика пока нет проверенных домашек.")

    homework_count = submissions[-1]["homework_number"]

    cached = await storage.get_cached_progress(student_id, homework_count)
    if cached is not None:
        return cached

    errors = await storage.list_errors(student_id)
    provider = os.getenv("AI_PROVIDER", "anthropic")
    try:
        if provider == "openai_compat":
            report = await analyze_progress_openai_compat(
                student_id, homework_count, errors
            )
        else:
            report = await analyze_progress(student_id, homework_count, errors)
    except json.JSONDecodeError:
        raise HTTPException(
            502, "Модель вернула некорректный ответ. Попробуй ещё раз."
        )
    except Exception as e:
        raise HTTPException(502, f"Ошибка обращения к модели ({provider}): {e}")

    await storage.save_progress(student_id, homework_count, report)
    return report
