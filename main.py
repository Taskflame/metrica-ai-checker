"""Точка входа прототипа: FastAPI + статичная страница для теста.

Запуск:
    uv run uvicorn main:app --reload
или:
    python -m uvicorn main:app --reload
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # подхватываем ANTHROPIC_API_KEY из .env ДО импорта сервиса

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from ai_checker import storage  # noqa: E402
from ai_checker.router import router as ai_router  # noqa: E402

app = FastAPI(title="МЕТРИКА — ИИ-проверка домашки (прототип)")
app.include_router(ai_router, prefix="/api")

STATIC_DIR = Path(__file__).parent / "static"


@app.on_event("startup")
async def on_startup() -> None:
    await storage.init_db()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", reload=True)


if __name__ == "__main__":
    main()
