"""Task 21's documentation contract, as decision D42 scopes it.

Plan §U Task 21 fixes three doc tests. D42 retains all three, and records that two
of them currently have nothing to inspect: the README publishes no metric table and
neither document publishes a table of the robustness probe's figures, because no
model has yet been evaluated on a real corpus.

A test that passes because there is nothing to look at proves nothing about its
checker. So each rule below is paired with tests that feed its checker a small
markdown snippet containing a violation and assert the violation is caught. Those
snippets are checker inputs, never published content, and they carry placeholder
cells rather than numbers: no figure is fabricated even as test data.

    A  every metric table in README.md has a baseline column
    B  no sentence naming the robustness probe says "transfers to",
       "generalises to" or "generalizes to" (decision D19), in either document
    C  any table presenting the robustness probe's figures carries its
       result_classification

Two further tests pin D42's current state — no metric table in the README, and no
probe figure table in either document. They are expected to change when a later
decision closes D42's deviation; the three rules above are not.
"""

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "phase-2-reproducibility.md"
DOCUMENTS = (README, GUIDE)

PROHIBITED = ("transfers to", "generalises to", "generalizes to")
"""Plan §U Task 21's three strings, from decision D19's binding prohibition."""

_METRIC_WORDS = (
    r"accuracy|precision|recall|recall@\w+|f1|macro[- ]?f1|pr[- ]?auc|roc[- ]?auc|auc"
    r"|top[- ]?k|metric|score"
)
METRIC = re.compile(rf"\b(?:{_METRIC_WORDS})\b", re.IGNORECASE)
"""A column header naming a model-quality measure makes a table a metric table."""

FIGURE = re.compile(
    rf"\b(?:{_METRIC_WORDS}|baseline|base rate|count|value|figure)\b", re.IGNORECASE
)
"""Wider than METRIC: a probe table presenting figures under any of these headers,
or with any of them as a row label, is presenting results."""

BASELINE = re.compile(r"\bbaseline\b", re.IGNORECASE)
CLASSIFICATION = re.compile(r"result[_ ]classification", re.IGNORECASE)
PROBE = re.compile(r"robustness probe|xdomain_xtarget_probe", re.IGNORECASE)
"""How a table's text, heading or caption identifies it as the probe's."""

SENTENCE_PROBE = re.compile(r"\bprobe\b", re.IGNORECASE)
"""Stricter for sentences: the prohibition must not be evaded by a short form."""

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*`(\[])")


# --- reading markdown --------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    line: int
    heading: str
    """The nearest markdown heading above the table."""
    caption: str
    """The paragraph immediately above the table."""
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def text(self) -> str:
        return " ".join((*self.header, *(cell for row in self.rows for cell in row)))


def _cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip().strip("`").strip() for cell in line.strip().strip("|").split("|"))


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and set(stripped) <= set("|-: ")


def _without_fences(markdown: str) -> list[str]:
    """The document's lines, with fenced code blocks blanked out."""
    lines, fenced = [], False
    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            lines.append("")
            continue
        lines.append("" if fenced else line)
    return lines


def tables(markdown: str) -> list[Table]:
    """Every pipe table in the document, outside fenced code."""
    lines = _without_fences(markdown)
    found: list[Table] = []
    heading = ""
    current: list[str] = []
    """The paragraph being read."""
    previous = ""
    """The last complete paragraph. Markdown separates a caption from its table with
    a blank line, so a blank line ends a paragraph without discarding it."""
    index = 0
    while index < len(lines):
        line = lines[index]
        is_row = line.strip().startswith("|")
        if is_row and index + 1 < len(lines) and _is_separator(lines[index + 1]):
            header = _cells(line)
            rows = []
            cursor = index + 2
            while cursor < len(lines) and lines[cursor].strip().startswith("|"):
                rows.append(_cells(lines[cursor]))
                cursor += 1
            caption = " ".join(current) if current else previous
            found.append(Table(index + 1, heading, caption, header, tuple(rows)))
            current, previous = [], ""
            index = cursor
            continue
        if line.startswith("#"):
            heading, current, previous = line.lstrip("#").strip(), [], ""
        elif not line.strip():
            if current:
                previous, current = " ".join(current), []
        else:
            current.append(line.strip())
        index += 1
    return found


def sentences(markdown: str) -> list[str]:
    """Sentences, outside fenced code; a table row counts as one sentence."""
    blocks: list[str] = []
    current: list[str] = []
    for line in _without_fences(markdown):
        if not line.strip() or line.strip().startswith("|"):
            if current:
                blocks.append(" ".join(current))
                current = []
            if line.strip().startswith("|"):
                blocks.append(line)
            continue
        current.append(line.strip())
    if current:
        blocks.append(" ".join(current))
    found = []
    for block in blocks:
        plain = re.sub(r"\s+", " ", block.replace("**", "").replace("`", ""))
        found.extend(part for part in SENTENCE_END.split(plain) if part.strip())
    return found


# --- the three rules, as checkers -------------------------------------------------------


def metric_tables(markdown: str) -> list[Table]:
    return [table for table in tables(markdown) if any(METRIC.search(h) for h in table.header)]


def metric_tables_without_baseline(markdown: str) -> list[Table]:
    """Rule A's violations."""
    return [
        table
        for table in metric_tables(markdown)
        if not any(BASELINE.search(h) for h in table.header)
    ]


def prohibited_probe_sentences(markdown: str) -> list[str]:
    """Rule B's violations."""
    return [
        sentence
        for sentence in sentences(markdown)
        if SENTENCE_PROBE.search(sentence)
        and any(phrase in sentence.lower() for phrase in PROHIBITED)
    ]


def probe_figure_tables(markdown: str) -> list[Table]:
    """Tables presenting the robustness probe's figures.

    A table is the probe's when its own text, its section heading or its caption
    names the probe, and it presents figures when a header or a row label is a
    figure word.
    """
    found = []
    for table in tables(markdown):
        is_probe = any(PROBE.search(part) for part in (table.text(), table.heading, table.caption))
        labels = (*table.header, *(row[0] for row in table.rows if row))
        if is_probe and any(FIGURE.search(label) for label in labels):
            found.append(table)
    return found


def probe_tables_without_classification(markdown: str) -> list[Table]:
    """Rule C's violations."""
    return [
        table for table in probe_figure_tables(markdown) if not CLASSIFICATION.search(table.text())
    ]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --- A. every metric table in the README has a baseline column ---------------------------


def test_every_readme_metric_table_has_a_baseline_column():
    """Plan §U Task 21, retained by D42."""
    offenders = metric_tables_without_baseline(read(README))
    assert not offenders, [f"line {table.line}: {table.header}" for table in offenders]


def test_the_baseline_rule_rejects_a_metric_table_without_one():
    """The guard for the deferred publication: a baseline-free metric table fails."""
    snippet = "## Results\n\n| Model | PR-AUC |\n| --- | --- |\n| triage | — |\n"
    assert len(metric_tables_without_baseline(snippet)) == 1


def test_the_baseline_rule_accepts_a_metric_table_with_one():
    snippet = "| Model | PR-AUC | Majority baseline |\n| --- | --- | --- |\n| triage | — | — |\n"
    assert metric_tables(snippet) and not metric_tables_without_baseline(snippet)


def test_the_baseline_rule_ignores_a_table_that_is_not_a_metric_table():
    snippet = "| Variable | Required | Purpose |\n| --- | --- | --- |\n| `SECRET_KEY` | Yes | — |\n"
    assert tables(snippet) and not metric_tables(snippet)


def test_the_readme_tables_are_actually_read():
    """Against a parser that silently found nothing: the README has a real table."""
    headers = [table.header for table in tables(read(README))]
    assert ("Variable", "Required", "Purpose") in headers


def test_the_readme_publishes_no_metric_table_under_d42():
    """D42's current state. Expected to change when a later decision closes D42."""
    assert not metric_tables(read(README))


# --- B. D19's prohibition, in every sentence naming the probe ------------------------------


def test_no_sentence_naming_the_probe_uses_prohibited_wording():
    """Plan §U Task 21 and decision D19, in both documents."""
    for path in DOCUMENTS:
        offenders = prohibited_probe_sentences(read(path))
        assert not offenders, (path.name, offenders)


def test_the_wording_rule_catches_a_violation():
    snippet = (
        "The reduced-feature cross-domain cross-target robustness probe generalises to CFPB.\n"
    )
    assert prohibited_probe_sentences(snippet)


def test_the_wording_rule_catches_a_short_form():
    """Naming the probe informally does not evade the prohibition."""
    for phrase in PROHIBITED:
        assert prohibited_probe_sentences(f"The probe {phrase} the other task.\n"), phrase


def test_the_wording_rule_is_scoped_to_sentences_naming_the_probe():
    """The contract's scope is the sentence, so a neighbouring sentence is not caught."""
    snippet = "The probe was measured. A different model transfers to another task.\n"
    assert not prohibited_probe_sentences(snippet)


def test_both_documents_exist_and_mention_the_probe():
    """Against a rule passing because a document was empty or missing."""
    for path in DOCUMENTS:
        assert any(SENTENCE_PROBE.search(sentence) for sentence in sentences(read(path))), path


# --- C. a probe figure table carries its result_classification ----------------------------


def test_any_probe_figure_table_carries_its_result_classification():
    """Plan §U Task 21, retained by D42, in both documents."""
    for path in DOCUMENTS:
        offenders = probe_tables_without_classification(read(path))
        assert not offenders, (path.name, [table.line for table in offenders])


def test_the_classification_rule_rejects_a_probe_table_without_one():
    """Premature publication without a classification is caught."""
    snippet = (
        "| Evaluation | Reduced-feature cross-domain cross-target robustness probe PR-AUC |\n"
        "| --- | --- |\n"
        "| cross-domain | — |\n"
    )
    assert len(probe_tables_without_classification(snippet)) == 1


def test_the_classification_rule_accepts_a_probe_table_with_one():
    snippet = (
        "| Evaluation | Robustness probe PR-AUC | result_classification |\n"
        "| --- | --- | --- |\n"
        "| cross-domain | — | — |\n"
    )
    assert probe_figure_tables(snippet) and not probe_tables_without_classification(snippet)


def test_a_probe_table_is_recognised_by_its_section_heading():
    """A table under the probe's heading is the probe's even if its cells never say so."""
    snippet = (
        "## The reduced-feature cross-domain cross-target robustness probe\n\n"
        "| Metric | In-domain | Cross-domain |\n| --- | --- | --- |\n| PR-AUC | — | — |\n"
    )
    assert len(probe_tables_without_classification(snippet)) == 1


def test_a_probe_table_is_recognised_by_a_row_label():
    """Figures listed as rows rather than columns are still figures."""
    snippet = (
        "The reduced-feature cross-domain cross-target robustness probe:\n\n"
        "| Evaluation | In-domain | Cross-domain |\n| --- | --- | --- |\n| PR-AUC | — | — |\n"
    )
    assert len(probe_tables_without_classification(snippet)) == 1


def test_no_probe_figure_table_is_published_under_d42():
    """D42's current state: the probe has published no real-corpus figure.

    Expected to change when a later decision closes D42's deviation.
    """
    for path in DOCUMENTS:
        assert not probe_figure_tables(read(path)), path.name
