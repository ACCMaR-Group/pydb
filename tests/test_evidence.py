import json
import shutil
import subprocess

import pytest

from pydb.evidence import normalize_latex_with_map, segment_source_text


def test_segment_keeps_abbreviation_text_and_covers_source():
    source = "Here, we define the ratio, i.e., alpha. Next."
    normalized, _ = normalize_latex_with_map(source)
    spans = segment_source_text(normalized)
    assert [span["text"] for span in spans] == ["Here, we define the ratio, i.e., alpha.", "Next."]
    assert "".join(span["text"] for span in spans).replace(" ", "") == normalized.replace(" ", "")


@pytest.mark.parametrize("source", [
    "😀 plain text.\n\nNext.",
    "A \\textbf{β😀} value.",
    "\ufeffA\u0085 B\t C\n\n\nD",
])
def test_bun_normalizer_and_span_parity(source):
    bun = shutil.which("bun")
    if not bun:
        pytest.skip("Bun is unavailable for cross-language parity check")
    evidence = "/home/gweng/codes/ai4s/sciturbo/src/science/evidence.ts"
    script = f"import {{normalizeLatexWithMap,segmentSourceText}} from '{evidence}'; const s={json.dumps(source)}; console.log(JSON.stringify({{normalized:normalizeLatexWithMap(s).normalized,maps:normalizeLatexWithMap(s).maps,spans:segmentSourceText(normalizeLatexWithMap(s).normalized)}}))"
    expected = json.loads(subprocess.check_output([bun, "-e", script], text=True))
    normalized, maps = normalize_latex_with_map(source)
    assert normalized == expected["normalized"]
    assert {"originalToNormalized": maps.originalToNormalized, "normalizedToOriginal": maps.normalizedToOriginal} == expected["maps"]
    actual_spans = segment_source_text(normalized)
    assert actual_spans == [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in expected["spans"]]
