from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asr_jetson.postprocessing import meeting_report  # noqa: E402


def _pandoc_available() -> bool:
    try:
        import pypandoc  # type: ignore
    except Exception:
        return False
    try:
        pypandoc.get_pandoc_version()  # type: ignore[attr-defined]
    except OSError:
        return False
    return True


def _sample_anonymized_markdown() -> str:
    return """### 3. THEMES DISCUSSED

1. **Invalidity pension refusal**
    - Grand Pont Industries refused the request without a detailed report.
    - Appeal deadline: 2 months from 30 September 2025 (due 30 November 2025).
    - No clear medical justification despite certificates being provided.
2. **Medical sequelae and work capacity**
    - Sequelae: stiffness in the right elbow, persistent pain, reduced mobility.
    - Associated issues: shoulder cyst, bursitis, cervical disc problems.
    - Impact on daily and professional life (computer use, carrying loads).
    - Uncertainty about whether recent certificates were considered.
"""


def test_deanonymize_report_preserves_list_indentation():
    mapping = {
        "pseudonym_reverse_map": {
            "Grand Pont Industries": "Region",
        }
    }
    anonymized_md = _sample_anonymized_markdown()
    restored = meeting_report.deanonymize_report_markdown(anonymized_md, mapping)

    bullet_lines = [
        line for line in restored.splitlines() if line.lstrip().startswith("- ")
    ]
    assert bullet_lines, "Expected bullet lines in the deanonymized report"
    assert all(
        line.startswith("    - ") for line in bullet_lines
    ), "Bullet indentation must be preserved for nested lists"


def test_pandoc_html_keeps_nested_list_structure():
    if not _pandoc_available():
        pytest.skip("pandoc is required to validate nested list HTML output")

    mapping = {
        "pseudonym_reverse_map": {
            "Grand Pont Industries": "Region",
        }
    }
    anonymized_md = _sample_anonymized_markdown()
    restored = meeting_report.deanonymize_report_markdown(anonymized_md, mapping)

    html = meeting_report._build_html_report(
        restored,
        title="Test",
        report_date="01/01/2025",
    )

    nested_list_re = re.compile(
        r"<ol[^>]*>.*?<li>.*?<ul[^>]*>.*?</ul>.*?</li>.*?</ol>",
        re.DOTALL,
    )
    assert nested_list_re.search(
        html
    ), "Expected nested ordered/unordered lists in the HTML output"


def test_deanonymize_report_keeps_context_first_name_for_unknown_full_name():
    anonymized_md = "Micheline Martin et Laura Blanc valident la décision."
    mapping = {
        "pseudonym_reverse_map": {
            "Laura Blanc": "Françoise",
        },
        "context_names": ["Micheline", "Françoise"],
    }

    restored = meeting_report.deanonymize_report_markdown(anonymized_md, mapping)

    assert "Micheline et Françoise valident la décision." in restored
    assert "Micheline Martin" not in restored


def test_deanonymize_report_restores_unique_first_name_alias_for_person() -> None:
    anonymized_md = "Hugo confirme le point. Manon valide la stratégie."
    mapping = {
        "entities": [
            {
                "tag": "<PERSON_1>",
                "type": "PERSON",
                "canonical": "M. Marlon",
                "pseudonym": "Hugo Durand",
                "mentions": ["M. Marlon"],
            },
            {
                "tag": "<PERSON_2>",
                "type": "PERSON",
                "canonical": "Micheline Martin",
                "pseudonym": "Manon Gauthier",
                "mentions": ["Micheline Martin"],
            },
            {
                "tag": "<ORGANIZATION_1>",
                "type": "ORGANIZATION",
                "canonical": "Astronov",
                "pseudonym": "Prisme Conseil",
                "mentions": ["Astronov"],
            },
        ],
        "pseudonym_map": {
            "<PERSON_1>": "Hugo Durand",
            "<PERSON_2>": "Manon Gauthier",
            "<ORGANIZATION_1>": "Prisme Conseil",
        },
        "pseudonym_reverse_map": {
            "Hugo Durand": "M. Marlon",
            "Manon Gauthier": "Micheline Martin",
            "Prisme Conseil": "Astronov",
        },
    }

    restored = meeting_report.deanonymize_report_markdown(anonymized_md, mapping)

    assert "M. Marlon confirme le point." in restored
    assert "Micheline Martin valide la stratégie." in restored


def test_deanonymize_report_does_not_restore_ambiguous_first_name_alias() -> None:
    anonymized_md = "Hugo Durand intervient. Ensuite, Hugo répond."
    mapping = {
        "pseudonym_reverse_map": {
            "Hugo Durand": "M. Marlon",
            "Hugo Martin": "M. Dupont",
        }
    }

    restored = meeting_report.deanonymize_report_markdown(anonymized_md, mapping)

    assert "M. Marlon intervient." in restored
    assert "Ensuite, Hugo répond." in restored
