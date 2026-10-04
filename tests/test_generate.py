# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz <109816464+FPGArtktic@users.noreply.github.com>
"""The parser, the note layout and the guards on the output folder.

One test per row of CLAUDE.md's rule table that says "enforced by: parser".
A rule that is not tested here is a rule the prompt only asks for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import config, generate, ollama
from app.chunk import Window

LABELS = {"p0042", "p0043", "p0044"}


@pytest.fixture
def window(tmp_path: Path) -> Window:
    text = tmp_path / "001.md"
    text.write_text(
        "[p0042] So the question everyone asks is why we cannot filter after "
        "sampling.\n"
        "[p0043] Let me write the condition on the board.\n"
        "[p0044] And that is the sampling theorem.\n",
        encoding="utf-8",
    )
    return Window(
        ord=3, start_s=760.0, end_s=1265.0, labels=tuple(sorted(LABELS)), text_path=text
    )


def _response(**overrides: str) -> str:
    sections = {
        "In short": "Filtering after sampling cannot undo aliasing. The alias "
        "sits in the same band as the signal.",
        "Explanation": "The speaker developed this from the sampling condition.",
        "Key claims": "- A filter cannot remove an alias. [p0042]\n"
        "- The condition was written on the board. [p0043]",
        "Formulas": "None in this window.",
        "Glossary": "- aliasing — aliasing\n"
        "- sampling rate — częstotliwość próbkowania",
        "Pitfalls": "- Do not assume a filter helps. [p0042]",
    }
    sections.update(overrides)
    return "\n\n".join(
        f"## {name}\n{body}" for name, body in sections.items() if body is not None
    )


def _parse(**overrides: str) -> generate.ParsedNote:
    return generate.parse(_response(**overrides), LABELS, 3)


# --- rules that reject the whole response -----------------------------------


def test_a_missing_required_section_fails() -> None:
    with pytest.raises(generate.NoteRejected, match="missing section: Explanation"):
        generate.parse(_response(Explanation=""), LABELS, 3)


def test_a_label_that_is_not_in_the_window_fails() -> None:
    """A reference that looks right and is not invalidates the whole note."""
    with pytest.raises(generate.NoteRejected, match="p0099"):
        _parse(**{"Key claims": "- An invented citation. [p0099]"})


def test_a_timestamp_anywhere_in_the_body_fails() -> None:
    """The model is never shown a time, so a time in its answer was invented."""
    with pytest.raises(generate.NoteRejected, match="timestamp"):
        _parse(Explanation="At 12:40 the speaker said this.")


def test_a_formulas_block_that_cites_no_label_fails() -> None:
    with pytest.raises(generate.NoteRejected, match="Formulas block"):
        _parse(Formulas="$$f_s > 2 f_{max}$$\nThe sampling condition.")


def test_a_formula_with_a_label_is_kept() -> None:
    parsed = _parse(Formulas="$$f_s > 2 f_{max}$$\nThe condition. [p0043]")
    assert "[[_Transcript#^p0043]]" in parsed.sections["Formulas"]


def test_none_in_this_window_needs_no_label() -> None:
    assert "None in this window." in _parse().sections["Formulas"]


def test_a_note_whose_every_claim_lost_its_label_fails() -> None:
    """Without one link into the transcript the note has no purpose left."""
    with pytest.raises(generate.NoteRejected, match="no claim"):
        _parse(**{"Key claims": "- A claim with no label.\n- Another one."})


# --- rules that cost one line -----------------------------------------------


def test_a_claim_without_a_label_is_dropped_and_counted() -> None:
    parsed = _parse(
        **{"Key claims": "- Labelled. [p0042]\n- Unlabelled, so discarded."}
    )
    assert parsed.dropped_claims == 1
    assert "Unlabelled" not in parsed.sections["Key claims"]
    assert "Labelled." in parsed.sections["Key claims"]


def test_a_claim_wrapped_over_lines_keeps_its_label() -> None:
    """The model wraps a long claim and puts the label on the last line."""
    parsed = _parse(
        **{
            "Key claims": "- A claim long enough that the model wrapped it\n"
            "  over a second line before the label. [p0042]"
        }
    )
    assert parsed.dropped_claims == 0
    assert "second line" in parsed.sections["Key claims"]


def test_a_glossary_line_in_the_wrong_shape_is_dropped_and_counted() -> None:
    parsed = _parse(Glossary="- aliasing — aliasing\n- a line with no separator at all")
    assert parsed.dropped_glossary == 1
    assert "no separator" not in parsed.sections["Glossary"]


def test_pitfalls_are_cited_like_claims() -> None:
    parsed = _parse(Pitfalls="- Unlabelled warning.")
    assert parsed.dropped_claims == 1
    assert parsed.sections["Pitfalls"] == ""


# --- labels become links ----------------------------------------------------


def test_a_label_becomes_a_block_reference() -> None:
    parsed = _parse()
    assert "([[_Transcript#^p0042]])" in parsed.sections["Key claims"]
    assert "[p0042]" not in parsed.sections["Key claims"]


def test_several_labels_on_one_claim_share_one_parenthesis() -> None:
    parsed = _parse(**{"Key claims": "- A claim resting on two. [p0042] [p0043]"})
    assert (
        "([[_Transcript#^p0042]] [[_Transcript#^p0043]])"
        in parsed.sections["Key claims"]
    )


# --- the rest of the parser -------------------------------------------------


def test_a_fenced_block_and_an_invented_heading_are_stripped() -> None:
    raw = f"```markdown\n# A title it was told not to write\n{_response()}\n```"
    parsed = generate.parse(raw, LABELS, 3)
    assert "A title it was told not to write" not in parsed.sections["In short"]
    assert parsed.sections["Explanation"]


def test_latex_delimiters_are_rewritten() -> None:
    parsed = _parse(Formulas=r"\[ f_s > 2 f_{max} \]" + "\nThe condition. [p0043]")
    assert "$$f_s > 2 f_{max}$$" in parsed.sections["Formulas"]
    assert "\\[" not in parsed.sections["Formulas"]


def test_the_title_comes_from_the_models_first_sentence() -> None:
    parsed = _parse()
    assert parsed.title == "Filtering after sampling cannot undo aliasing"


def test_a_long_first_sentence_is_cut_at_a_word() -> None:
    parsed = _parse(
        **{
            "In short": "This window is about a subject whose description runs "
            "well past any reasonable length for a file name on disk."
        }
    )
    assert len(parsed.title) <= generate.TITLE_MAX_CHARS
    assert not parsed.title.endswith(" ")
    assert " who" in parsed.title


def test_a_title_never_carries_a_path_separator() -> None:
    """The output folder may be synced to a filesystem less forgiving than ext4."""
    parsed = _parse(**{"In short": 'Input/output impedance: the "high" one.'})
    assert "/" not in parsed.title
    assert ":" not in parsed.title
    assert '"' not in parsed.title


# --- the note ---------------------------------------------------------------


def _note(window: Window, **overrides: str) -> str:
    return generate.assemble(
        window,
        generate.parse(_response(**overrides), LABELS, window.ord),
        source_name="lecture-04.mp4",
        slug="lec4",
        domain="signal processing",
        model="qwen2.5:32b",
    )


def test_the_note_matches_the_output_contract(window: Window) -> None:
    note = _note(window)
    assert note.startswith("---\ngenerator: tachygraf\n")
    assert f"prompt: {config.PROMPT_VERSION}" in note
    assert 'source: "lecture-04.mp4"' in note
    assert 'span: "00:12:40 - 00:21:05"' in note
    assert "model: qwen2.5:32b" in note
    assert "tags: [signal-processing, lec4]" in note
    assert "# 03 Filtering after sampling cannot undo aliasing" in note


def test_the_sections_come_out_in_the_contract_order(window: Window) -> None:
    note = _note(window)
    found = [line[3:] for line in note.splitlines() if line.startswith("## ")]
    assert found == list(generate.SECTION_ORDER)


def test_an_empty_section_is_left_out(window: Window) -> None:
    note = _note(window, Pitfalls="")
    assert "## Pitfalls" not in note
    assert "## Glossary" in note


def test_a_note_is_recognised_as_ours(window: Window) -> None:
    assert generate.is_ours_text(_note(window))


def test_a_file_without_the_marker_is_not_ours() -> None:
    assert not generate.is_ours_text("---\ntitle: mine\n---\n# Notes\n")
    assert not generate.is_ours_text("# Just a heading\n")
    # The marker counts only inside the frontmatter block.
    assert not generate.is_ours_text("---\na: b\n---\ngenerator: tachygraf\n")


# --- the guards (copied from glosator) --------------------------------------


def test_a_path_outside_the_output_folder_is_refused() -> None:
    with pytest.raises(PermissionError, match="outside the output folder"):
        generate.check_inside(config.OUT_DIR / ".." / "escaped.md")


def test_an_existing_foreign_file_is_never_overwritten() -> None:
    foreign = config.OUT_DIR / "someone-elses.md"
    foreign.write_text("important\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="not written by tachygraf"):
        generate.write_note(foreign, "replaced\n")
    assert foreign.read_text(encoding="utf-8") == "important\n"


def test_our_own_file_is_replaced(window: Window) -> None:
    path = config.OUT_DIR / "01 Ours.md"
    generate.write_note(path, _note(window))
    generate.write_note(path, _note(window, Pitfalls=""))
    assert "## Pitfalls" not in path.read_text(encoding="utf-8")


def test_claim_dir_leaves_a_non_empty_unstamped_directory_alone() -> None:
    folder = config.OUT_DIR / "lec4"
    folder.mkdir()
    (folder / "the users own file.md").write_text("mine\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="did not write"):
        generate.claim_dir(folder)


def test_claim_dir_is_rerunnable_on_its_own_directory() -> None:
    folder = config.OUT_DIR / "lec4"
    generate.claim_dir(folder)
    (folder / "01 A note.md").write_text("x\n", encoding="utf-8")
    assert generate.claim_dir(folder) == folder.resolve()


def test_write_atomic_leaves_no_temporary_file_behind() -> None:
    path = config.OUT_DIR / "x.md"
    generate.write_atomic(path, "done\n")
    assert path.read_text(encoding="utf-8") == "done\n"
    assert list(config.OUT_DIR.glob("*.tmp")) == []


# --- the stage --------------------------------------------------------------


def test_run_writes_the_note_and_asks_the_model_once(
    window: Window, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[str] = []

    def fake_generate(model: str, prompt: str) -> str:
        prompts.append(prompt)
        return _response()

    monkeypatch.setattr(ollama, "generate", fake_generate)
    path = generate.run(
        window,
        slug="lec4",
        source_name="lecture-04.mp4",
        windows=8,
        domain="signal processing",
    )

    assert len(prompts) == 1, "the model is never asked twice in the same job"
    assert "[p0042]" in prompts[0], "the window text reaches the model with its labels"
    assert "Window: 3 of 8" in prompts[0]
    assert "00:12:40" not in prompts[0], "the model is never shown a time"
    assert path.name == "03 Filtering after sampling cannot undo aliasing.md"
    assert path.parent == config.OUT_DIR / "lec4"


def test_run_refuses_a_window_that_cannot_fit_the_context_window(
    window: Window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window.text_path.write_text(
        "[p0042] " + "word " * (config.CHUNK_MAX_TOKENS * config.CHARS_PER_TOKEN),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ollama, "generate", lambda *_: pytest.fail("the model must not be asked")
    )
    with pytest.raises(ValueError, match="exceed the"):
        generate.run(window, slug="lec4", source_name="lecture-04.mp4", windows=1)


def test_run_lets_a_rejection_through_with_the_rule_it_broke(
    window: Window, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ollama, "generate", lambda *_: _response(**{"Key claims": "- No label here."})
    )
    with pytest.raises(generate.NoteRejected, match="no claim"):
        generate.run(window, slug="lec4", source_name="lecture-04.mp4", windows=1)


def test_the_prompt_template_takes_exactly_the_fields_we_pass() -> None:
    """A renamed placeholder would otherwise surface on the first real run."""
    rendered = generate.render(
        generate.load_prompt("note.md"),
        domain="signal processing",
        source="lecture-04.mp4",
        window=3,
        windows=8,
        text="[p0042] Something was said.",
    )
    assert "{{" not in rendered
    for heading in generate.SECTION_ORDER:
        assert f"## {heading}" in rendered


# --- the title (the pilot's finding) ------------------------------------------

PILOT_FAILURE = (
    "This part of the lecture explains what diodes are, how they work, and "
    "their applications in controlling current flow direction in circuits."
)


def test_the_models_own_title_is_used() -> None:
    parsed = _parse(Title="Why a filter cannot undo aliasing")
    assert parsed.title == "Why a filter cannot undo aliasing"
    assert not parsed.title_substituted


def test_the_pilot_failure_no_longer_becomes_a_file_name() -> None:
    """What the 32B actually produced: a sentence about the lecture, cut short.

    It became the file name "01 This part of the lecture explains what diodes
    are, how they" and then appeared in every glossary wikilink.
    """
    parsed = _parse(**{"In short": PILOT_FAILURE})
    assert parsed.title == "What diodes are, how they work"
    assert not parsed.title.startswith("This part")
    assert parsed.title_substituted


def test_a_title_that_names_the_task_is_replaced() -> None:
    parsed = _parse(Title="This window explains the sampling theorem")
    assert not parsed.title.startswith("This window")
    assert parsed.title_substituted


def test_a_title_naming_the_speaker_is_replaced() -> None:
    parsed = _parse(Title="The speaker explains aliasing")
    assert parsed.title_substituted


def test_a_long_title_is_cut_at_a_clause_not_mid_phrase() -> None:
    long = (
        "Aliasing in sampled systems, and why a filter afterwards cannot "
        "remove it from the band"
    )
    parsed = _parse(Title=long)
    assert len(parsed.title) <= generate.TITLE_MAX_CHARS
    assert parsed.title == "Aliasing in sampled systems"


def test_a_cut_title_does_not_end_on_a_dangling_word() -> None:
    parsed = _parse(
        Title="How the emitter follower behaves when the load impedance is low"
    )
    assert len(parsed.title) <= generate.TITLE_MAX_CHARS
    assert parsed.title.split()[-1].lower() not in generate.DANGLING_WORDS


def test_a_missing_title_falls_back_and_is_counted() -> None:
    """A note is ten minutes of generation; a bad title does not waste it."""
    parsed = _parse()
    assert parsed.title == "Filtering after sampling cannot undo aliasing"
    assert parsed.title_substituted


def test_the_title_is_not_rendered_as_a_section(window: Window) -> None:
    note = _note(window, Title="Why a filter cannot undo aliasing")
    assert "## Title" not in note
    assert "# 03 Why a filter cannot undo aliasing" in note


def test_the_prompt_asks_for_a_title_and_forbids_a_calqued_glossary() -> None:
    """Both rules were added because a pilot run produced the failure."""
    prompt = generate.load_prompt("note.md")
    assert "## Title" in prompt
    assert "obszar zubożony" in prompt
    assert "brak odpowiednika" in prompt
