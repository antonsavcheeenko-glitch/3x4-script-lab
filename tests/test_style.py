from app.parsers import parse_file
from app.services.style_analyzer import build_profile, classify_ending, classify_intro, profile_to_prompt, text_metrics
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
    assert "Середня довжина речення" in prompt
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
