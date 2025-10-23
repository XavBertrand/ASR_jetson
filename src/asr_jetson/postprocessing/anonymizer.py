"""
Modern anonymisation utilities backed by local Hugging Face NER models and
optional Ollama-based quality control. The module exposes a functional API
suited for the pipeline: use ``anonymize_text`` to scrub a document and
``deanonymize_text`` to restore the content from the generated mapping.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

import re
try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

if requests is not None:  # pragma: no cover - trivial branch
    _RequestException = requests.RequestException  # type: ignore[attr-defined]
else:  # pragma: no cover - optional dependency path
    class _RequestException(Exception):
        pass
try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline  # type: ignore
except ImportError:  # pragma: no cover - optional heavy dependency
    AutoModelForTokenClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex "noyau dur" used for deterministic structured PII detection
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?){2,5}\d{2,4}")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.I)
SIREN_SIRET_RE = re.compile(r"\b(?:\d{9}|\d{14})\b")
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[\/\-\.:]\d{1,2}[\/\-\.:]\d{2,4}"
    r"|\d{4}[\/\-\.:]\d{1,2}[\/\-\.:]\d{1,2}"
    r"|(?:\d{1,2}\s+)?(?:janv\.?|févr\.?|mars|avr\.?|mai|juin|juil\.?|août|sept\.?|oct\.?|nov\.?|déc\.?)\s+\d{2,4})\b",
    re.I,
)

STRUCTURED_REGEX_SPECS: Sequence[Tuple[str, re.Pattern]] = [
    ("EMAIL", EMAIL_RE),
    ("PHONE", PHONE_RE),
    ("IBAN", IBAN_RE),
    ("SIREN_SIRET", SIREN_SIRET_RE),
    ("DATE", DATE_RE),
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def normalize_ollama_base_url(url: str) -> str:
    """
    Ensure the Ollama base URL does not contain duplicated path segments such as `/v1`.
    """
    raw = (url or "http://localhost:11434").strip()
    if not raw:
        raw = "http://localhost:11434"
    parsed = urlparse(raw)
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    normalized = parsed._replace(path=path, params="", query="", fragment="")
    base = urlunparse(normalized).rstrip("/")
    return base or "http://localhost:11434"


def _default_device() -> int:
    return 0 if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available() else -1


@dataclass
class Settings:
    """Runtime settings controlling NER thresholds, devices, and LLM QC."""

    model_id: str = "Jean-Baptiste/camembert-ner"
    per_threshold: float = 0.65
    org_threshold: float = 0.65
    loc_threshold: float = 0.65
    other_threshold: float = 0.7
    stride: int = 64
    device: int = field(default_factory=_default_device)
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "mistral:7b-instruct-q4_K_M"
    ollama_timeout: int = 30
    enable_llm_qc: bool = True

    def __post_init__(self) -> None:
        self.ollama_base_url = normalize_ollama_base_url(self.ollama_base_url)
        self.llm_model = (self.llm_model or "mistral").strip()


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------
SENT_SPLIT_RE = re.compile(r"(?<=\.|\?|!|\n)\s+")


def normalize_text(s: str) -> str:
    """Reduce common Unicode variants and enforce spacing after titles."""
    s = (
        s.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2212", "-")
        .replace("’", "'")
        .replace("‘", "'")
    )
    s = re.sub(r"\b(Mme|Mlle|Me|Ma(?:ître|itre)|Dr|Pr|M\.)\s?(?=[A-Za-zÀ-ÖØ-öø-ÿ])", r"\1 ", s)
    return s


def split_sentences(text: str) -> List[str]:
    parts = SENT_SPLIT_RE.split(text)
    return [p for p in parts if p.strip()]


def make_blocks(text: str, sentences: Sequence[str], max_chars: int = 1200, max_sents: int = 5) -> List[Dict]:
    """Group sentences into blocks while preserving offsets."""
    blocks: List[Dict] = []
    cursor = 0
    sent_spans: List[Tuple[int, int]] = []
    for sentence in sentences:
        pos = text.find(sentence, cursor)
        if pos == -1:
            pos = text.find(sentence)
            if pos == -1:
                continue
        sent_spans.append((pos, pos + len(sentence)))
        cursor = pos + len(sentence)

    i = 0
    while i < len(sent_spans):
        block_start = sent_spans[i][0]
        block_end = block_start
        count = 0
        k = i
        while k < len(sent_spans) and count < max_sents:
            block_end = sent_spans[k][1]
            count += 1
            if block_end - block_start >= max_chars:
                k += 1
                break
            k += 1
        blocks.append({"start": block_start, "end": block_end, "text": text[block_start:block_end]})
        i = k
    return blocks


# ---------------------------------------------------------------------------
# Hugging Face NER
# ---------------------------------------------------------------------------
_PIPELINE_CACHE: Dict[Tuple[str, int, int], Any] = {}


def load_ner_pipeline(cfg: Settings):
    if AutoTokenizer is None or AutoModelForTokenClassification is None or pipeline is None:
        raise RuntimeError(
            "transformers is required to load the NER pipeline; provide a custom 'ner_pipeline' when testing"
        )
    key = (cfg.model_id, cfg.device, cfg.stride)
    if key not in _PIPELINE_CACHE:
        tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        mdl = AutoModelForTokenClassification.from_pretrained(cfg.model_id)
        _PIPELINE_CACHE[key] = pipeline(
            "token-classification",
            model=mdl,
            tokenizer=tok,
            aggregation_strategy="simple",
            device=cfg.device,
            # stride=cfg.stride,
            # truncation=True,
        )
    return _PIPELINE_CACHE[key]


def spans_from_ner(ner_outputs: List[Dict], cfg: Settings) -> List[Dict]:
    spans: List[Dict] = []
    for item in ner_outputs:
        ent = str(item.get("entity_group", "")).upper()
        score = float(item.get("score", 0.0))
        start = int(item.get("start", -1))
        end = int(item.get("end", -1))
        if start < 0 or end <= start:
            continue

        keep = False
        typ = ent
        if ent in {"PER", "B-PER", "I-PER"}:
            keep = score >= cfg.per_threshold
            typ = "PER"
        elif ent in {"ORG", "B-ORG", "I-ORG"}:
            keep = score >= cfg.org_threshold
            typ = "ORG"
        elif ent in {"LOC", "B-LOC", "I-LOC"}:
            keep = score >= cfg.loc_threshold
            typ = "LOC"
        else:
            keep = score >= cfg.other_threshold

        if keep:
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "type": typ,
                    "score": score,
                    "source": "ner",
                }
            )
    return spans


def spans_from_regex(text: str) -> List[Dict]:
    spans: List[Dict] = []
    for typ, pattern in STRUCTURED_REGEX_SPECS:
        for match in pattern.finditer(text):
            a, b = match.span()
            if b > a:
                spans.append(
                    {
                        "start": a,
                        "end": b,
                        "type": typ,
                        "score": 1.0,
                        "source": "regex",
                    }
                )
    return spans


def spans_from_catalog(
    text: str,
    entries: Optional[Iterable[Dict]],
    *,
    as_person: bool = False,
    default_label: str = "CAT",
) -> List[Dict]:
    if not entries:
        return []

    spans: List[Dict] = []
    for entry in entries:
        pattern = (
            entry.get("pattern")
            or entry.get("text")
            or entry.get("value")
            or entry.get("label")
        )
        if not pattern:
            continue
        raw_label = str(entry.get("label") or default_label or "CAT").upper()
        label = "PER" if as_person else raw_label
        try:
            regex = re.compile(re.escape(pattern), re.I | re.M)
        except re.error as exc:
            logger.warning("Invalid catalog pattern %s: %s", pattern, exc)
            continue
        for match in regex.finditer(text):
            a, b = match.span()
            spans.append(
                {
                    "start": a,
                    "end": b,
                    "type": label,
                    "score": 0.95,
                    "source": "catalog",
                    "catalog_pattern": pattern,
                }
            )
    return spans


def merge_and_dedup_spans(spans: List[Dict]) -> List[Dict]:
    if not spans:
        return []
    seen = set()
    filtered: List[Dict] = []
    for span in spans:
        key = (span["start"], span["end"], span["type"])
        if key not in seen:
            seen.add(key)
            filtered.append(span)

    def _priority(src: str, typ: str) -> int:
        if typ in {"PER", "ORG", "LOC"}:
            return 0 if src == "ner" else 1
        return 0 if src == "regex" else 1

    filtered.sort(
        key=lambda s: (s["start"], -(s["end"] - s["start"]), _priority(s.get("source", ""), s["type"]), -s.get("score", 0.0))
    )

    result: List[Dict] = []
    for span in filtered:
        if not result:
            result.append(span)
            continue
        prev = result[-1]
        if span["start"] < prev["end"]:
            len_prev = prev["end"] - prev["start"]
            len_span = span["end"] - span["start"]
            if len_span > len_prev:
                result[-1] = span
        else:
            result.append(span)
    return result


# def apply_masks(text: str, spans: List[Dict]) -> Tuple[str, List[Dict]]:
#     tag_counter: Dict[str, int] = {}
#     span_indices = sorted(enumerate(spans), key=lambda item: item[1]["start"], reverse=True)
#     anonymized = text
#     for idx, span in span_indices:
#         span["original"] = text[span["start"] : span["end"]]
#         typ = span["type"].upper()
#         tag_counter.setdefault(typ, 0)
#         tag_counter[typ] += 1
#         tag = f"<{typ}_{tag_counter[typ]}>"
#         anonymized = anonymized[: span["start"]] + tag + anonymized[span["end"] :]
#         spans[idx]["tag"] = tag
#     return anonymized, spans



def apply_masks(text: str, spans: List[Dict]) -> Tuple[str, List[Dict]]:
     tag_counter: Dict[str, int] = {}
     span_indices = sorted(enumerate(spans), key=lambda item: item[1]["start"], reverse=True)
     anonymized = text
     for idx, span in span_indices:
         # --- Trim des espaces qui entourent le span pour ne pas "manger" l'espace avant ---
         a = int(span["start"])
         b = int(span["end"])
         # Rogne les espaces à gauche
         while a < b and a < len(text) and text[a].isspace():
             a += 1
         # Rogne les espaces à droite
         while b > a and b - 1 < len(text) and text[b - 1].isspace():
             b -= 1
         # Si tout a été rogné (span vide), on saute
         if b <= a:
             continue
         # Met à jour le span
         span["start"], span["end"] = a, b
         span["original"] = text[a:b]
         typ = span["type"].upper()
         tag_counter.setdefault(typ, 0)
         tag_counter[typ] += 1
         tag = f"<{typ}_{tag_counter[typ]}>"
         anonymized = anonymized[:a] + tag + anonymized[b:]
         spans[idx]["tag"] = tag
     return anonymized, spans



def collect_spans_by_blocks(text: str, blocks: Sequence[Dict], ner_pipe, cfg: Settings) -> List[Dict]:
    spans: List[Dict] = []
    for block in blocks:
        block_text = block["text"]
        base = block["start"]
        ner_out = ner_pipe(block_text)
        ner_spans = spans_from_ner(ner_out, cfg)
        for span in ner_spans:
            span["start"] += base
            span["end"] += base
        rx_spans = spans_from_regex(block_text)
        for span in rx_spans:
            span["start"] += base
            span["end"] += base
        spans.extend(ner_spans + rx_spans)
    return merge_and_dedup_spans(spans)


# ---------------------------------------------------------------------------
# Ollama-assisted QC
# ---------------------------------------------------------------------------
def call_ollama_chat(
    base_url: str,
    model: str,
    anonymized_block: str,
    *,
    timeout: int = 30,
) -> List[Dict]:
    """
    Call the Ollama /v1/chat/completions endpoint asking for additional spans
    that should be anonymised. Returns local spans (start/end relative to the
    provided block).
    """
    if requests is None:
        raise RuntimeError("The 'requests' package is required for Ollama quality-control")

    base = normalize_ollama_base_url(base_url)
    url = f"{base}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    system_prompt = (
        "Tu es un vérificateur d'anonymisation. "
        "On te fournit un texte déjà anonymisé (balises <PER_#>, <ORG_#>, etc.). "
        "Détecte les mentions PERSONNES/ORGANISATIONS/LIEUX qui n'ont pas été masquées. "
        "Réponds uniquement en JSON: liste de spans "
        '[{"start": int, "end": int, "type": "PER|ORG|LOC"}].'
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Texte anonymisé:\n{anonymized_block}\n\nRéponds en JSON.",
            },
        ],
        "temperature": 0.0,
        "top_p": 1.0,
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    try:
        raw_spans = json.loads(content)
    except Exception:
        logger.warning("Ollama QC returned non-JSON payload; ignoring.")
        return []

    cleaned: List[Dict] = []
    for span in raw_spans:
        try:
            start = int(span["start"])
            end = int(span["end"])
            typ = str(span["type"]).upper()
        except (KeyError, ValueError, TypeError):
            continue
        if start < 0 or end <= start or typ not in {"PER", "ORG", "LOC"}:
            continue
        cleaned.append({"start": start, "end": end, "type": typ, "score": 0.99, "source": "llm"})
    return cleaned


def qc_llm_extend_mapping(
    anonymized_text: str,
    blocks_original: Sequence[Dict],
    mapping_spans: List[Dict],
    cfg: Settings,
) -> Tuple[str, List[Dict]]:
    """
    Slide a window over the already anonymised text and call Ollama to
    discover residual entities. Newly detected spans are masked and added
    to the mapping.
    """
    new_text = anonymized_text

    counters: Dict[str, int] = {}
    for span in mapping_spans:
        tag = span.get("tag")
        if not tag:
            continue
        m = re.match(r"<([A-Z_]+)_(\d+)>", tag)
        if m:
            counters[m.group(1)] = max(counters.get(m.group(1), 0), int(m.group(2)))

    pos = 0
    extended: List[Dict] = []

    for block in blocks_original:
        approx_len = block["end"] - block["start"]
        window_start = pos
        window_end = min(len(new_text), pos + int(approx_len * 1.4) + 400)
        block_text = new_text[window_start:window_end]

        try:
            additional_spans = call_ollama_chat(
                cfg.ollama_base_url,
                cfg.llm_model,
                block_text,
                timeout=cfg.ollama_timeout,
            )
        except (RuntimeError, _RequestException) as exc:
            logger.warning("Ollama QC failed (%s); continuing without extra masking.", exc)
            return new_text, mapping_spans

        if additional_spans:
            for span in sorted(additional_spans, key=lambda s: s["start"], reverse=True):
                a = max(0, min(window_start + span["start"], len(new_text)))
                b = max(0, min(window_start + span["end"], len(new_text)))
                if b <= a:
                    continue
                typ = span["type"]
                counters[typ] = counters.get(typ, 0) + 1
                tag = f"<{typ}_{counters[typ]}>"
                original_segment = new_text[a:b]
                new_text = new_text[:a] + tag + new_text[b:]
                extended.append(
                    {
                        "type": typ,
                        "tag": tag,
                        "start": a,
                        "end": a + len(tag),
                        "source": "llm",
                        "score": span.get("score", 0.99),
                        "original": original_segment,
                    }
                )
            pos = min(len(new_text), window_start + max(int(approx_len * 0.9), 1))
        else:
            pos = min(len(new_text), window_start + max(approx_len, 1))

    if extended:
        mapping_spans.extend(extended)
    return new_text, mapping_spans


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------
TYPE_CANON = {
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "LOC": "LOCATION",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "IBAN": "IBAN",
    "SIREN_SIRET": "SIREN_SIRET",
    "DATE": "DATE",
}


def _unique(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_mapping(spans: List[Dict], original_text: str, cfg: Settings) -> Dict:
    tags: Dict[str, Dict] = {}
    summary: Dict[str, int] = {}

    for span in spans:
        tag = span.get("tag")
        if not tag:
            continue
        typ_raw = span.get("type", "").upper()
        typ = TYPE_CANON.get(typ_raw, typ_raw or "UNKNOWN")

        entry = tags.setdefault(
            tag,
            {
                "type": typ,
                "mentions": [],
                "sources": set(),
                "scores": [],
            },
        )

        mention = span.get("original")
        if not mention:
            mention = original_text[span["start"] : span["end"]]
        if mention:
            entry["mentions"].append(mention)
        entry["sources"].add(span.get("source", "unknown"))
        if span.get("score") is not None:
            entry["scores"].append(float(span["score"]))
        summary[typ] = summary.get(typ, 0) + 1

    entities: List[Dict] = []
    tag_lookup: Dict[str, str] = {}

    for tag, data in sorted(tags.items(), key=lambda item: item[0]):
        mentions = _unique(m.strip() for m in data["mentions"] if m)
        mentions = [m for m in mentions if m]
        canonical = max(mentions, key=lambda val: len(val), default=mentions[0] if mentions else "")
        replacement = canonical or (mentions[0] if mentions else tag)

        entity = {
            "tag": tag,
            "type": data["type"],
            "canonical": replacement,
            "mentions": mentions,
            "sources": sorted(data["sources"]),
        }
        if data["scores"]:
            entity["score_avg"] = round(sum(data["scores"]) / len(data["scores"]), 4)
        entities.append(entity)
        tag_lookup[tag] = replacement

    return {
        "entities": entities,
        "summary": summary,
        "tag_lookup": tag_lookup,
        "meta": {
            "model_id": cfg.model_id,
            "ollama_base_url": cfg.ollama_base_url,
            "llm_model": cfg.llm_model,
            "enable_llm_qc": cfg.enable_llm_qc,
        },
    }


def build_tag_lookup(mapping: Dict, restore: str = "canonical") -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for entity in mapping.get("entities", []):
        tag = entity.get("tag")
        if not tag:
            continue
        mentions = entity.get("mentions") or []
        if restore == "first" and mentions:
            replacement = mentions[0]
        elif restore == "longest" and mentions:
            replacement = max(mentions, key=len)
        else:
            replacement = entity.get("canonical") or (mentions[0] if mentions else tag)
        lookup[tag] = replacement
    if not lookup and mapping.get("tag_lookup"):
        lookup.update(mapping["tag_lookup"])
    return lookup


def deanonymize_text(anonymized_text: str, mapping: Dict, restore: str = "canonical") -> str:
    lookup = build_tag_lookup(mapping, restore=restore)
    if not lookup:
        return anonymized_text
    pattern = re.compile("|".join(re.escape(k) for k in sorted(lookup, key=len, reverse=True)))
    return pattern.sub(lambda match: lookup.get(match.group(0), match.group(0)), anonymized_text)


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------
def load_catalog(path: str | Path, default_label: str = "CAT") -> List[Dict]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    text = path_obj.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path_obj.suffix.lower() in {".json", ".jsonl"}:
        data = json.loads(text)
        if isinstance(data, dict):
            entries = [{"pattern": k, "label": v} for k, v in data.items()]
        elif isinstance(data, list):
            entries = data
        else:
            logger.warning("Unsupported catalog JSON format at %s", path)
            entries = []
    else:
        entries = [
            {"pattern": line.strip(), "label": default_label}
            for line in text.splitlines()
            if line.strip()
        ]
    return entries


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------
def anonymize_text(
    raw_text: str,
    *,
    settings: Optional[Settings] = None,
    ner_pipeline=None,
    catalog_entries: Optional[Iterable[Dict]] = None,
    catalog_default_label: str = "CAT",
    catalog_as_person: bool = False,
    max_block_chars: int = 1200,
    max_block_sents: int = 5,
) -> Tuple[str, Dict]:
    cfg = settings or Settings()
    text = normalize_text(raw_text or "")
    sentences = split_sentences(text)
    blocks = make_blocks(text, sentences, max_chars=max_block_chars, max_sents=max_block_sents)

    ner = ner_pipeline or load_ner_pipeline(cfg)

    spans = collect_spans_by_blocks(text, blocks, ner, cfg)
    spans.extend(spans_from_catalog(text, catalog_entries, as_person=catalog_as_person, default_label=catalog_default_label))
    spans = merge_and_dedup_spans(spans)
    anonymized_text, mapping_spans = apply_masks(text, spans)

    if cfg.enable_llm_qc and cfg.ollama_base_url:
        if requests is None:
            logger.warning("requests is not installed; skipping Ollama QC phase.")
        else:
            anonymized_text, mapping_spans = qc_llm_extend_mapping(anonymized_text, blocks, mapping_spans, cfg)

    mapping = build_mapping(mapping_spans, text, cfg)
    return anonymized_text, mapping


# ---------------------------------------------------------------------------
# CLI entry point for standalone usage
# ---------------------------------------------------------------------------
def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Anonymise a text file and output its mapping.")
    parser.add_argument("-i", "--input", required=True, help="Input text file (UTF-8).")
    parser.add_argument("-o", "--output-text", help="Output anonymised text file.")
    parser.add_argument("-m", "--output-mapping", help="Output mapping JSON file.")
    parser.add_argument("--ollama-url", default=None, help="Override Ollama base URL (default: http://localhost:11434).")
    parser.add_argument("--ollama-model", default=None, help="Override Ollama model name (default: mistral).")
    parser.add_argument("--disable-llm", action="store_true", help="Skip Ollama quality-control pass.")
    parser.add_argument("--catalog", help="Optional gazetteer (JSON or newline-separated text).")
    parser.add_argument("--catalog-label", default="CAT", help="Default label for plain-text catalog entries.")
    parser.add_argument("--catalog-as-person", action="store_true", help="Treat catalog entries as PERSON entities.")
    parser.add_argument("--max-block-chars", type=int, default=1200, help="Maximum characters per block.")
    parser.add_argument("--max-block-sents", type=int, default=5, help="Maximum sentences per block.")
    args = parser.parse_args(argv)

    raw_text = Path(args.input).read_text(encoding="utf-8")
    cfg = Settings()
    if args.ollama_url:
        cfg.ollama_base_url = args.ollama_url
    if args.ollama_model:
        cfg.llm_model = args.ollama_model
    if args.disable_llm:
        cfg.enable_llm_qc = False

    catalog_entries = None
    if args.catalog:
        catalog_entries = load_catalog(args.catalog, default_label=args.catalog_label)

    anonymized_text, mapping = anonymize_text(
        raw_text,
        settings=cfg,
        catalog_entries=catalog_entries,
        catalog_default_label=args.catalog_label,
        catalog_as_person=args.catalog_as_person,
        max_block_chars=args.max_block_chars,
        max_block_sents=args.max_block_sents,
    )

    output_text_path = Path(args.output_text) if args.output_text else Path(args.input).with_suffix(".anon.txt")
    output_mapping_path = Path(args.output_mapping) if args.output_mapping else Path(args.input).with_suffix(".anon.json")

    output_text_path.write_text(anonymized_text, encoding="utf-8")
    output_mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Anonymisation done. Tags: %s", mapping.get("summary"))
    logger.info("Anonymised text → %s", output_text_path)
    logger.info("Mapping JSON    → %s", output_mapping_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
