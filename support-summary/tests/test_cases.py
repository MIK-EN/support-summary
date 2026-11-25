from datetime import datetime
from pathlib import Path

from summary.adapters import load_cases
from summary.llm import load_cache
from summary.render import render_full
from summary.rules import evaluate_all, load_rules

PROJECT_DIR = Path(__file__).resolve().parent.parent
GOLDEN = """# Сводка support · 23.11.2026 10:00

🔴 2 · 🟡 1 · ⚪ 0 · 🟢 1

🔴 **1. П-08 · Поставка карт · срок уже нарушен**
**CRM:** доставка сегодня до 10:00, статус «В пути».
**Чат (не подтверждено), курьер:** «Задерживаюсь, раньше 13:00 не приеду»
⚠️ Расхождение: по CRM 10:00, по словам курьера 13:00.
→ Предупредить точку, подтвердить новое время, обновить срок в CRM.

🔴 **2. С-24 · Увольнение сотрудника · срок сегодня**
**CRM:** последний рабочий день сегодня, подтверждение руководителя отсутствует.
**Чат (не подтверждено), сотрудник:** «С руководителем всё обсудили, он в курсе»
⚠️ В чате утверждается, что сделано. Слова подтверждением не считаются.
→ Сегодня получить подтверждение руководителя в CRM.

🟡 **3. С-17 · Оформление сотрудника · срок завтра**
**CRM:** первый выход завтра, документы ожидаются.
**Чат (не подтверждено), сотрудник:** «Вчера отправил фотографии документов руководителю в личку»
⚠️ В чате утверждается, что сделано. Слова подтверждением не считаются.
→ Запросить документы у руководителя, загрузить в CRM сегодня.

🟢 **4. П-09 · Поставка карт · принято по CRM**
**CRM:** статус «Принято». Приёмка подтверждена сегодня 09:20.
**Чат (не подтверждено), получатель:** «Карты получили, количество проверили»
→ Действий не требуется.
"""


def test_render_full_matches_golden_summary_for_employer_cases() -> None:
    now = datetime(2026, 11, 23, 10, 0)
    cases = load_cases(PROJECT_DIR / "data" / "cases.json")
    cache = load_cache(PROJECT_DIR / "data" / "llm_cache.json")
    rules = load_rules(PROJECT_DIR / "rules.yaml")

    assert render_full(evaluate_all(cases, cache, now, rules), now) == GOLDEN
