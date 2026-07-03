"""Pydantic-схемы результата проверки домашнего задания.

Эти же схемы можно перенести в metrica-backend как есть —
они не зависят от остального кода прототипа.
"""

from pydantic import BaseModel, Field


class MathError(BaseModel):
    """Одна найденная ошибка."""

    location: str = Field(description="Где ошибка: номер задания/строка/фрагмент")
    student_wrote: str = Field(description="Что написал ученик")
    correct: str = Field(description="Как должно быть")
    explanation: str = Field(description="Почему это ошибка, понятным языком")
    topic: str = Field(description="Тема, к которой относится ошибка")
    severity: str = Field(description="conceptual | computational | careless")


class TaskReview(BaseModel):
    """Разбор одного задания с фото."""

    task_label: str = Field(description="Номер или краткое условие задания")
    transcription: str = Field(description="Решение ученика, распознанное с фото")
    is_correct: bool
    errors: list[MathError] = []

    # Поля режима ЕГЭ (в обычном режиме остаются пустыми)
    ege_task_number: int | None = Field(
        default=None, description="Номер задания ЕГЭ (1-19), если распознан"
    )
    max_score: int | None = Field(
        default=None, description="Максимальный первичный балл за задание"
    )
    estimated_score: int | None = Field(
        default=None, description="Оценка по критериям ФИПИ"
    )
    score_rationale: str | None = Field(
        default=None, description="Обоснование оценки в терминах критериев"
    )
    justification_gaps: list[str] = Field(
        default_factory=list,
        description="Пробелы в обосновании, за которые снимут баллы "
        "(даже при верном ответе)",
    )


class PracticeTask(BaseModel):
    """Задание для работы над ошибками."""

    topic: str
    question: str
    hint: str = Field(description="Подсказка, если ученик застрял")
    answer: str = Field(description="Ответ для самопроверки")
    solution: str = Field(description="Краткое решение")


class HomeworkReview(BaseModel):
    """Полный результат работы ИИ-агента."""

    readable: bool = Field(description="Удалось ли разобрать фото")
    quality_note: str | None = Field(
        default=None, description="Замечание к качеству фото, если есть"
    )
    tasks: list[TaskReview] = []
    summary: str = Field(default="", description="Общий вывод для ученика")
    error_topics: list[str] = Field(
        default_factory=list, description="Темы, в которых были ошибки"
    )
    practice: list[PracticeTask] = Field(
        default_factory=list, description="Задания для работы над ошибками"
    )
