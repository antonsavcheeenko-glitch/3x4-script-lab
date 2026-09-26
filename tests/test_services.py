import json

import pytest

from app.db.models import ExtractedFact
from app.llm import LLMNotConfigured, LLMService, extract_json
from app.services import drafting, editing, outline, projects, research, style_service
from app.services.fact_check import analyze_paragraph, analyze_with_llm, classify_sentence
from tests.conftest import SAMPLES

REPORT = (SAMPLES / "research" / "zvit_utrymannya_gravtsiv.md").read_bytes()


def test_create_project_and_validation(session):
    p = projects.create_project(session, "Тест", "тема", "опис")
    assert p.id and p.created_at
    assert projects.list_projects(session)[0].id == p.id
    with pytest.raises(ValueError):
        projects.create_project(session, "  ")


def test_ingest_and_heuristic_facts(session):
    p = projects.create_project(session, "Казино")
    doc = research.ingest_document(session, p, "zvit.md", REPORT, save_file=False)
    assert doc.source_type == "report"
    assert doc.chunks and doc.chunks[0].start_char == 0
    facts, method = research.extract_facts(session, p, doc)
    assert method == "heuristic"
    stmts = " ".join(f.statement for f in facts)
    assert "62%" in stmts and "47 хвилин" in stmts
    assert all(f.chunk_id for f in facts)
    assert all(0 < f.confidence <= 1 for f in facts)


def test_reextract_keeps_starred_and_manual(session):
    p = projects.create_project(session, "Казино")
    doc = research.ingest_document(session, p, "zvit.md", REPORT, save_file=False)
    facts, _ = research.extract_facts(session, p, doc)
    facts[0].starred = True
    session.add(ExtractedFact(project_id=p.id, document_id=doc.id, statement="Ручний факт 2025 року про 5 платформ.", method="manual"))
    session.commit()
    research.extract_facts(session, p, doc)
    session.refresh(doc)
    assert any(f.method == "manual" for f in doc.facts)
    assert any(f.id == facts[0].id for f in doc.facts)
    # без дублікатів
    stmts = [f.statement for f in doc.facts]
    assert len(stmts) == len(set(stmts))


def test_guess_source_type():
    assert research.guess_source_type("interview_andriy.txt", "") == "interview"
    assert research.guess_source_type("x.txt", "Закон України. Стаття 5. Постанова суду. Ухвала.") == "legal"
    assert research.guess_source_type("x.txt", "просто текст") == "unknown"


def test_demo_load_and_outline(session):
    p = projects.load_demo(session)
    assert len(p.documents) == 3
    assert p.style_profile and p.style_profile.profile["metrics"]
    assert len(p.outline) == 9
    assert [b.position for b in p.outline] == list(range(9))
    assert sum(len(b.fact_ids) for b in p.outline) >= 8
    # переміщення блоку
    blocks = sorted(p.outline, key=lambda b: b.position)
    outline.move_block(session, p, blocks[1], -1)
    assert sorted(p.outline, key=lambda b: b.position)[0].id == blocks[1].id


def test_template_draft_without_llm(session):
    p = projects.load_demo(session)
    draft = drafting.get_or_create_draft(session, p)
    out = drafting.generate_full(session, p, draft, llm=None)
    assert len(out) == 9
    assert all(b.method == "template" for b in out)
    used = {i for b in out for i in b.used_fact_ids}
    assert used
    assert all(f.used_in_script == (f.id in used) for f in p.facts)
    md = drafting.export_markdown(p, draft)
    assert "Джерела тверджень" in md and "[F" in md
    assert "[F" not in drafting.export_markdown(p, draft, with_citations=False)


def test_llm_draft_citations_and_variants(session, fake_llm):
    p = projects.load_demo(session)
    fid = p.facts[0].id

    def responder(system, prompt):
        assert "ПРОФІЛЬ СТИЛЮ" in system  # профіль стилю підставляється
        return f"Перше речення з фактом [F{fid}]. Друге без джерела. Вигаданий [F99999]."

    llm = fake_llm(responder)
    draft = drafting.get_or_create_draft(session, p)
    block = sorted(p.outline, key=lambda b: b.position)[0]
    v1 = drafting.generate_block(session, p, draft, block, llm)
    assert v1.method == "llm" and v1.used_fact_ids == [fid]  # неіснуючий F99999 відкинуто
    v2 = drafting.generate_block(session, p, draft, block, llm, alternative=True)
    assert v2.variant == 2 and v2.is_selected and not v1.is_selected
    assert "АЛЬТЕРНАТИВНА" in llm.provider.calls[-1][1]
    drafting.select_variant(session, draft, v1)
    assert v1.is_selected and not v2.is_selected
    drafting.update_block_text(session, p, v1, "Текст без маркерів.")
    assert v1.used_fact_ids == []
    assert not next(f for f in p.facts if f.id == fid).used_in_script


def test_llm_fact_extraction(session, fake_llm):
    p = projects.create_project(session, "Казино")
    doc = research.ingest_document(session, p, "zvit.md", REPORT, save_file=False)
    llm = fake_llm(lambda s, pr: "```json\n" + json.dumps([
        {"statement": "62% користувачів заходять 4+ рази на тиждень", "snippet": "62% активних користувачів онлайн-казино", "category": "number", "confidence": 0.8},
        {"statement": "Погана категорія", "snippet": "", "category": "???", "confidence": "abc"},
    ], ensure_ascii=False) + "\n```")
    facts, method = research.extract_facts(session, p, doc, llm)
    assert method == "llm" and len(facts) == 2
    assert facts[0].chunk_id is not None and facts[0].confidence == 0.8
    assert facts[1].category == "other" and facts[1].confidence == 0.6


def test_llm_outline_generation(session, fake_llm):
    p = projects.load_demo(session)
    fid = p.facts[0].id
    llm = fake_llm(lambda s, pr: [
        {"block_type": "hook", "title": "Лист із бонусом", "notes": "н", "target_words": 120, "fact_ids": [fid, 999999]},
        {"block_type": "weird", "title": "Щось", "notes": "", "target_words": "abc"},
    ])
    blocks = outline.generate_outline(session, p, llm)
    assert len(blocks) == 2
    assert blocks[0].fact_ids == [fid]
    assert blocks[1].block_type == "custom" and blocks[1].target_words == 250


def test_style_profile_service(session, fake_llm):
    prof = style_service.create_profile(session, "Я")
    for f in sorted((SAMPLES / "style").glob("*.md")):
        style_service.add_corpus_file(session, prof, f.name, f.read_bytes())
    style_service.rebuild_profile(session, prof)
    assert prof.profile["corpus"]["documents"] == 3
    llm = fake_llm(lambda s, pr: {"voice": "Сухо і точно.", "do": ["цифри"], "avoid": ["пафос"], "sample_sentences": ["Це методика."]})
    style_service.enrich_with_llm(session, prof, llm)
    assert prof.profile["llm_notes"]["voice"] == "Сухо і точно."
    assert "Сухо і точно." in style_service.profile_prompt(prof)
    # перерахунок зберігає LLM-нотатки
    style_service.rebuild_profile(session, prof)
    assert prof.profile["llm_notes"]["voice"] == "Сухо і точно."


def test_edit_operations(session, fake_llm):
    p = projects.create_project(session, "x")
    out, method, op_id = editing.run_operation(session, p, "reduce_cliches", "Варто зазначити, що ринок зріс. У сучасному світі це важливо.", None)
    assert method == "heuristic"
    assert "зазначити" not in out.lower() and out.startswith("Ринок")
    with pytest.raises(RuntimeError):
        editing.run_operation(session, p, "shorten", "Текст.", None)
    llm = fake_llm(lambda s, pr: "Коротко.")
    out, method, op_id = editing.run_operation(session, p, "shorten", "Довгий текст.", llm)
    assert out == "Коротко." and method == "llm"
    editing.mark_applied(session, op_id)
    session.refresh(p)
    assert len(p.edits) == 2 and any(e.applied for e in p.edits)


def test_fact_vs_interpretation_heuristic():
    assert classify_sentence("Хіба це нормально?")[0] == "rhetorical"
    assert classify_sentence("Можливо, регулятор не встигає.")[0] == "speculation"
    assert classify_sentence("За даними звіту, 62% гравців заходять щодня.")[0] == "fact"
    assert classify_sentence("На мою думку, це системна проблема.")[0] == "interpretation"
    facts = [ExtractedFact(id=1, statement="Середня тривалість сесії становить 47 хвилин.", snippet="", confidence=0.6)]
    res = analyze_paragraph("Казино створені, щоб люди втрачали контроль. Середня тривалість сесії становить 90 хвилин.", facts)
    assert res[0].too_strong and "розроблені так, щоб" in res[0].suggestion
    assert res[1].too_strong and res[1].supporting_fact_ids == [1] and "47" in res[1].suggestion


def test_fact_vs_interpretation_llm(fake_llm):
    llm = fake_llm(lambda s, pr: {"sentences": [{"sentence": "Казино створені, щоб люди втрачали контроль.", "type": "interpretation", "too_strong": True, "issue": "узагальнення", "suggestion": "Багато механік казино розроблені, щоб збільшувати тривалість гри.", "supporting_fact_ids": ["x"]}]})
    res = analyze_with_llm("Казино створені, щоб люди втрачали контроль.", [], llm)
    assert res[0].too_strong and res[0].suggestion.startswith("Багато") and res[0].supporting_fact_ids == []


def test_llm_service_not_configured_and_json():
    svc = LLMService(None)
    assert not svc.available
    with pytest.raises(LLMNotConfigured):
        svc.complete("s", "p", purpose="t")
    assert extract_json('Ось: ```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('текст [1, 2] текст') == [1, 2]
