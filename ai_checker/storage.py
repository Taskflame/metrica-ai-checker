"""SQLite-персистентность: домашки, ошибки, кэш сводок прогресса ученика.

Первая часть проекта, которой нужна персистентность — до сих пор ai_checker
был полностью stateless. student_id — свободный текст, вводимый вручную
(см. README): при переносе в metrica-backend заменяется на реальный user_id
из системы аккаунтов, остальная логика не меняется.
"""

import os
from datetime import datetime, timezone

import aiosqlite

from .schemas import HomeworkReview, ProgressReport, TaskReview
from .taxonomy import TASK_TYPE_LABELS

DB_PATH = os.getenv("AI_CHECKER_DB_PATH", "metrica_ai_checker.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    homework_number INTEGER NOT NULL,
    homework_label TEXT NOT NULL,
    mode TEXT NOT NULL,
    summary TEXT NOT NULL,
    review_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_submissions_student ON submissions(student_id);

CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL REFERENCES submissions(id),
    student_id TEXT NOT NULL,
    homework_label TEXT NOT NULL,
    task_label TEXT NOT NULL,
    topic TEXT NOT NULL,
    severity TEXT NOT NULL,
    explanation TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_errors_student ON errors(student_id);

CREATE TABLE IF NOT EXISTS progress_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    up_to_number INTEGER NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_progress_student_upto
    ON progress_reports(student_id, up_to_number);
"""


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_SCHEMA)
        await db.commit()


def compute_homework_label(number: int, tasks: list[TaskReview]) -> str:
    """Собирает ярлык вида "дз3: задачи+уравнения" по типам заданий,
    проставленным моделью. Порядок — по TASK_TYPE_LABELS, не алфавитный."""
    present = {t for task in tasks for t in task.task_types}
    ordered = [label for key, label in TASK_TYPE_LABELS.items() if key in present]
    if not ordered:
        ordered = [TASK_TYPE_LABELS["other"]]
    return f"дз{number}: {'+'.join(ordered)}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def save_submission(
    student_id: str, mode: str, review: HomeworkReview
) -> tuple[int, str]:
    """Сохраняет проверенную домашку и её ошибки одной транзакцией.
    Возвращает (homework_number, homework_label)."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM submissions WHERE student_id = ?", (student_id,)
        )
        row = await cursor.fetchone()
        homework_number = row[0] + 1
        homework_label = compute_homework_label(homework_number, review.tasks)
        created_at = _now()

        cursor = await db.execute(
            """INSERT INTO submissions
               (student_id, homework_number, homework_label, mode, summary, review_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                student_id,
                homework_number,
                homework_label,
                mode,
                review.summary,
                review.model_dump_json(),
                created_at,
            ),
        )
        submission_id = cursor.lastrowid

        error_rows = [
            (
                submission_id,
                student_id,
                homework_label,
                task.task_label,
                error.topic,
                error.severity,
                error.explanation,
                created_at,
            )
            for task in review.tasks
            for error in task.errors
        ]
        if error_rows:
            await db.executemany(
                """INSERT INTO errors
                   (submission_id, student_id, homework_label, task_label, topic, severity, explanation, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                error_rows,
            )

        await db.commit()

    return homework_number, homework_label


async def list_submissions(student_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT homework_number, homework_label, mode, summary, created_at
               FROM submissions WHERE student_id = ? ORDER BY homework_number""",
            (student_id,),
        )
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def list_errors(student_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT homework_label, task_label, topic, severity, explanation
               FROM errors WHERE student_id = ? ORDER BY id""",
            (student_id,),
        )
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def get_cached_progress(student_id: str, up_to_number: int) -> ProgressReport | None:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT report_json FROM progress_reports WHERE student_id = ? AND up_to_number = ?",
            (student_id, up_to_number),
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    return ProgressReport.model_validate_json(row[0])


async def save_progress(student_id: str, up_to_number: int, report: ProgressReport) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO progress_reports (student_id, up_to_number, report_json, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(student_id, up_to_number) DO UPDATE SET report_json = excluded.report_json""",
            (student_id, up_to_number, report.model_dump_json(), _now()),
        )
        await db.commit()
