from app.parsers import parse_file
from app.services.style_analyzer import recurring_formulas, build_profile, classify_ending, classify_intro, profile_to_prompt, text_metrics
from app.services.style_check import check_style
from tests.conftest import SAMPLES


def corpus():
    return [parse_file(p).text for p in sorted((SAMPLES / "style").glob("*.md"))]


def test_text_metrics_basic():
    m = text_metrics("Я прийшов. Мені 34 роки. Хіба це важливо?\n\nНовий абзац тут. Але є нюанс.")
    assert m["sentences"] == 5
    assert m["paragraphs"] == 2
    assert m["question_rate"] == 0.2
    assert m["author_i_per_1k"] > 0
    assert m["numbers_per_1k"] > 0
    assert m["transitions"].get("але") == 1


def test_build_profile_structure():
    p = build_profile(corpus(), name="Тест")
    for key in ("metrics", "tone", "do", "avoid", "intro_style", "ending_style", "transition_style", "lexicon", "summary"):
        assert key in p
    m = p["metrics"]
    assert p["corpus"]["documents"] == 3
    assert 3 < m["sentence_len"]["mean"] < 20
    assert m["sentence_len"]["p25"] <= m["sentence_len"]["median"] <= m["sentence_len"]["p75"]
    assert m["numbers_per_1k"] > 10  # демо-корпус щільний на цифри
    assert m["cliches_per_1k"] < 1
    assert p["lexicon"]["bigrams"], "мають бути повторювані біграми"
    assert "варто зазначити" in p["rare_or_absent_cliches"]


def test_profile_prompt_contains_rules():
    prompt = profile_to_prompt(build_profile(corpus()))
    assert "РОБИ" in prompt and "УНИКАЙ" in prompt
    assert "Типова довжина речення" in prompt
    assert profile_to_prompt({}).startswith("Профіль стилю не задано")


def test_intro_and_ending_classification():
    assert classify_intro("У цьому відео ми розберемо казино.") == "generic"
    assert classify_intro("14 березня 2021 року Андрій зробив перший депозит.") == "scene_date"
    assert classify_intro("Чому батареї холодні?") == "question"
    assert classify_ending("Підписуйтесь на канал і ставте лайк.") == "cta"
    assert classify_ending("Отже, система працює саме так.") == "summary"


def test_style_check_distinguishes_good_and_bad():
    prof = build_profile(corpus())
    bad = (
        "У сучасному світі варто зазначити, що проблема відіграє важливу роль у суспільстві, оскільки відповідальність, "
        "стабільність і ефективність функціонування інституцій є невід'ємною частиною розвитку держави в умовах трансформації. "
        "Це неймовірно важливо! Давайте розберемося."
    )
    good = "Кілограм бурштину коштує від 1 000 до 3 000 доларів. Копач ризикує найбільше. А заробляє найменше."
    rb, rg = check_style(bad, prof), check_style(good, prof)
    assert rg.score > rb.score + 40
    msgs = " ".join(d.message for d in rb.diagnostics)
    assert "Кліше" in msgs and "абстрактних" in msgs and "Шаблонний вступ" in msgs


def test_style_check_without_profile_uses_default():
    r = check_style("Короткий текст. З цифрою 5.", None)
    assert r.used_default_reference
    assert 0 <= r.score <= 100


def test_recurring_formulas_merge_overlaps():
    sign = "Мене звати Антон це канал три на чотири й ми вирушаємо на пошуки альтернативної думки."
    tails = ["Потім історія про вугілля.", "Сьогодні розберемо алгоритми.", "Отже, почнемо з фактів."]
    texts = [f"Вступ про тему {i}. {sign} {tails[i]}" for i in range(3)]
    texts.append("Зовсім інший текст без підпису.")
    f = dict(recurring_formulas(texts))
    assert "мене звати антон це канал три на чотири й ми вирушаємо на пошуки альтернативної думки" in f
    # фрагменти, що перекриваються, склеєні — окремих шматків підпису немає
    assert not any(k != "мене звати антон це канал три на чотири й ми вирушаємо на пошуки альтернативної думки" and "вирушаємо" in k for k in f)


def test_long_lines_are_paragraphs_and_phrase_metric():
    from app.utils.text import split_paragraphs

    line = "Це довгий абзац, який написано одним рядком без переносів, як у файлах з Google Docs, " * 4
    text = f"{line}\n{line}\n{line}"
    assert len(split_paragraphs(text)) == 3
    assert len(split_paragraphs("короткий\nрядок\nверстки")) == 1
    m = text_metrics("Перша фраза, друга фраза, третя фраза. Ще одна.")
    assert m["phrase_len"]["mean"] == 2.0


def test_own_phrases_not_penalized():
    base = "Давайте розберемося, як це працює. У 2020 році було 5 випадків. "
    prof = build_profile([base * 5, base * 5])
    r = check_style("Давайте розберемося, як це працює. У 2021 році було 7 випадків.", prof)
    assert not any("Кліше" in d.message for d in r.diagnostics)
