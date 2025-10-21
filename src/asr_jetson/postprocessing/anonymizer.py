"""
Anonymisation utilities used to detect and scrub PII with NER, gazetteers, and regexes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from flashtext import KeywordProcessor
from rapidfuzz import fuzz, process
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
from unidecode import unidecode


# --- Configuration constants ---
DEFAULT_MODEL = "cmarkea/distilcamembert-base-ner"
GAZ_FUZZY_THRESHOLD = 90
ALIAS_SIM_THRESHOLD = 82  # Slightly lower to group variants such as Xav <-> Xavier.
MAX_NGRAM_WORDS = 5
ENTITY_CANON = {"PER": "NOM", "ORG": "ORG", "LOC": "LIEU", "MISC": "CAT"}

# French stopwords used to filter fuzzy matches and avoid false positives.
STOPWORDS_FR = {
    "a", "à", "au", "aux", "chez", "de", "des", "du", "et", "en", "la", "le", "les", "un", "une",
    "ce", "cet", "cette", "sur", "dans", "par", "pour", "plus", "moins", "ou", "où", "avec",
    "sans", "se", "ses", "son", "sa", "leurs", "leur",
}

REGEX_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
REGEX_TEL = re.compile(r"(?<!\w)(?:\+?\d[\s\-.]?){9,}(?!\w)")
REGEX_TITLE_PERSON = re.compile(
    r"\b(?:M|Mr|Monsieur|Mme|Madame|Mlle|Mademoiselle)\.?\s+[A-ZÉÈÀÂÎÔÙÜÇ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
)

NICK_EQUIV = {
    "xav": "xavier",
    "alex": "alexandre",
    "fred": "frederic",
    "nico": "nicolas",
    "ben": "benjamin",
}

# Title + surname pattern (covers optional punctuation and expanded honorifics).
REGEX_TITLE_PERSON = re.compile(
    r"\b(?:M|Mr|Monsieur|Mme|Madame|Mlle|Mademoiselle)\.?\s+"
    r"[A-ZÉÈÀÂÎÔÙÜÇ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
)

_WORD_CHARS = r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]"

# Protect speaker headers such as "SPEAKER_1 :" (whitespace-insensitive).
REGEX_SPEAKER_HEADER = re.compile(r"(?m)^\s*SPEAKER_\d+\s*:\s*")

# Technical terms to keep as-is (no anonymisation to avoid over-scrubbing).
DENYLIST_EXACT = {"sms", "whatsapp", "mail", "email", "appel", "appeler"}


def _detect_ner_spans_chunked(text: str, ner_pipe: Any, max_chars: int = 2000, overlap: int = 80) -> List["Span"]:
    """
    Run the NER pipeline on sliding windows of ``text`` and re-map offsets globally.

    :param text: Raw text to analyse.
    :type text: str
    :param ner_pipe: HuggingFace pipeline callable producing NER spans.
    :type ner_pipe: Any
    :param max_chars: Maximum window size in characters.
    :type max_chars: int
    :param overlap: Number of characters to overlap between consecutive windows.
    :type overlap: int
    :returns: List of detected spans normalised to the original text offsets.
    :rtype: List[Span]
    """
    spans: List[Span] = []
    L = len(text)
    if L == 0:
        return spans

    start = 0
    while start < L:
        end = min(L, start + max_chars)
        # Avoid splitting in the middle of a word when possible.
        if end < L:
            # Walk back to a logical separator.
            j = end
            while j > start and text[j - 1].isalnum():
                j -= 1
            if j > start + int(0.7 * max_chars):
                end = j  # Only adjust if the chunk remains reasonably sized.
        chunk = text[start:end]

        # Execute NER on the current window.
        out = ner_pipe(chunk, aggregation_strategy="simple")

        # Recalculate global offsets and expand to word boundaries.
        for e in out:
            raw_lbl = e.get("entity_group") or e.get("entity", "")
            lbl = ENTITY_CANON.get(raw_lbl, raw_lbl)
            if lbl not in ("NOM", "ORG", "LIEU", "CAT"):
                continue
            a_local, b_local = int(e["start"]), int(e["end"])
            a_glob = start + a_local
            b_glob = start + b_local
            a_glob, b_glob = _expand_to_word_boundaries(text, a_glob, b_glob)
            mention = text[a_glob:b_glob]
            if lbl == "CAT" and _alnum_len(mention) < 3:
                continue
            spans.append(Span(mention, a_glob, b_glob, "PER" if lbl == "NOM" else lbl, "ner", 50))

        # Advance with overlap.
        if end >= L:
            break
        start = end - overlap if end - overlap > start else end

    return spans


def _post_label_corrections(spans: List["Span"], text: str) -> List["Span"]:
    """
    Apply heuristic label corrections to NER spans.

    :param spans: Raw spans produced by detectors.
    :type spans: List[Span]
    :param text: Original text content for contextual checks.
    :type text: str
    :returns: Filtered and relabelled span list.
    :rtype: List[Span]
    """
    corrected = []
    per_texts = [s.text for s in spans if s.label == "PER"]

    for s in spans:
        # Denylist.
        if s.text.lower() in DENYLIST_EXACT:
            continue

        # Context "chez X" implies the span is likely an organisation.
        ctx_start = max(0, s.start - 10)
        context = text[ctx_start:s.start].lower()
        if "chez" in context and s.label in ("PER", "CAT"):
            s = Span(s.text, s.start, s.end, "ORG", s.source, s.priority)

        # Acronyms (guarded against common polite words).
        if re.fullmatch(r"[A-Z]{3,8}", s.text) and s.text not in {"URGENT", "MERCI", "BONJOUR"}:
            s = Span(s.text, s.start, s.end, "ORG", s.source, s.priority)

        # ORG → PER when very similar to a known person.
        if s.label == "ORG" and re.match(r"^[A-ZÉ][a-zà-ÿ]+$", s.text):
            if per_texts:
                best = process.extractOne(s.text, per_texts, scorer=fuzz.WRatio)
                if best and best[1] >= 85:
                    s = Span(s.text, s.start, s.end, "PER", s.source, s.priority)

        corrected.append(s)
    return corrected


def _protected_ranges(text: str) -> List[Tuple[int, int]]:
    """
    Return the spans corresponding to speaker headers that must remain untouched.

    :param text: Text containing potential speaker headers.
    :type text: str
    :returns: List of ``(start, end)`` tuples.
    :rtype: List[Tuple[int, int]]
    """
    return [(m.start(), m.end()) for m in REGEX_SPEAKER_HEADER.finditer(text)]


def _drop_spans_in_protected(spans: List["Span"], protected_ranges: List[Tuple[int, int]]) -> List["Span"]:
    """
    Remove spans that overlap protected regions.

    :param spans: Candidate spans.
    :type spans: List[Span]
    :param protected_ranges: Intervals that should not be anonymised.
    :type protected_ranges: List[Tuple[int, int]]
    :returns: Filtered span list excluding overlaps with protected ranges.
    :rtype: List[Span]
    """
    if not protected_ranges:
        return spans
    kept = []
    for s in spans:
        if any(not (s.end <= a or s.start >= b) for a, b in protected_ranges):
            # Overlaps a protected zone -> drop it.
            continue
        kept.append(s)
    return kept


def _expand_to_word_boundaries(text: str, start: int, end: int) -> Tuple[int, int]:
    """
    Extend a character span to cover full word boundaries.

    :param text: Source text.
    :type text: str
    :param start: Inclusive start offset.
    :type start: int
    :param end: Exclusive end offset.
    :type end: int
    :returns: Tuple with expanded ``(start, end)`` offsets.
    :rtype: Tuple[int, int]
    """
    a, b = start, end
    while a > 0 and re.match(_WORD_CHARS, text[a - 1]):
        a -= 1
    while b < len(text) and re.match(_WORD_CHARS, text[b:b + 1]):
        b += 1
    return a, b


def _alnum_len(s: str) -> int:
    """
    Count the alphanumeric characters in ``s`` (including accented letters).

    :param s: Input string.
    :type s: str
    :returns: Number of alphanumeric characters.
    :rtype: int
    """
    return len(re.sub(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ]", "", s))


# French nicknames (add more over time as needed).


def _extract_email_name_sets(text: str) -> List[Set[str]]:
    """
    Extract token sets derived from email local parts.

    :param text: Text containing email addresses.
    :type text: str
    :returns: List of token sets sourced from email usernames.
    :rtype: List[Set[str]]
    """
    sets: List[Set[str]] = []
    for m in REGEX_EMAIL.finditer(text):
        local = m.group().split("@", 1)[0]
        toks = [unidecode(t).lower() for t in re.split(r"[._\-+]", local) if t]
        if toks:
            sets.append(set(toks))
    return sets


def _extended_name_tokens(s: str) -> Set[str]:
    """
    Produce normalised tokens and nickname variants for better email matching.

    :param s: Name string to normalise.
    :type s: str
    :returns: Set of normalised tokens including nickname substitutions.
    :rtype: Set[str]
    """
    toks = [t for t in re.split(r"[\s\-]+", _normalize_name(s)) if t]
    out = set(toks)
    for t in list(out):
        if t in NICK_EQUIV:
            out.add(NICK_EQUIV[t])
    return out


def _nickname_normalize(s: str) -> str:
    """
    Normalize nicknames to their canonical form when known.

    :param s: Raw nickname string.
    :type s: str
    :returns: Normalised nickname.
    :rtype: str
    """
    base = re.sub(r"[^a-z]", "", _normalize_name(s))
    return NICK_EQUIV.get(base, base)

def _strip_titles(s: str) -> str:
    """
    Remove French honorific titles from the provided name.

    :param s: Input string potentially containing titles.
    :type s: str
    :returns: String stripped of titles.
    :rtype: str
    """
    return re.sub(r"\b(mr|mme|mlle|m|m\.|mme\.|mlle\.)\b", "", s, flags=re.I)

def _normalize_name(s: str) -> str:
    """
    Normalise a name by removing accents, punctuation, and collapsing whitespace.

    :param s: Input name string.
    :type s: str
    :returns: Normalised lowercase name.
    :rtype: str
    """
    s = unidecode(s)
    s = _strip_titles(s)
    s = re.sub(r"[^\w\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _last_and_initials(s: str) -> Tuple[str, str]:
    """
    Extract the last token and initials from a normalised name string.

    :param s: Input name string.
    :type s: str
    :returns: Tuple containing ``(last_name, initials)``.
    :rtype: Tuple[str, str]
    """
    toks = [t for t in re.split(r"[\s\-]+", _normalize_name(s)) if t]
    if not toks:
        return "", ""
    last = toks[-1]
    initials = "".join(t[0] for t in toks)
    return last, initials

@dataclass
class Span:
    """Span metadata describing an anonymisable entity."""

    text: str
    start: int
    end: int
    label: str     # PER/ORG/LOC/EMAIL/TEL/CAT
    source: str    # ner/gazetteer_exact/gazetteer_fuzzy/regex
    priority: int

class Gazetteer:
    """Keyword-based gazetteer capable of exact and fuzzy span detection."""

    def __init__(self, entries: List[Tuple[str, str]]) -> None:
        """
        Build a gazetteer from ``(pattern, label)`` pairs.

        :param entries: Pairs of text patterns and their associated labels.
        :type entries: List[Tuple[str, str]]
        """
        self.entries = entries
        self.kp_by_label: Dict[str, KeywordProcessor] = {}
        for pat, label in entries:
            kp = self.kp_by_label.setdefault(label, KeywordProcessor())
            kp.add_keyword(pat, pat)
        self.entries_by_label: Dict[str, List[str]] = {}
        for pat, label in entries:
            self.entries_by_label.setdefault(label, []).append(pat)

    @staticmethod
    def load(path: str, default_label: str = "CAT") -> "Gazetteer":
        """
        Load gazetteer entries from a JSON or text file.

        :param path: Path to the gazetteer file.
        :type path: str
        :param default_label: Fallback label applied when the source lacks one.
        :type default_label: str
        :returns: Instantiated gazetteer populated with entries.
        :rtype: Gazetteer
        :raises ValueError: If the JSON file has an unexpected structure.
        """
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries: List[Tuple[str, str]] = []
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    for item in data:
                        entries.append((item["pattern"], item.get("label", default_label)))
                else:
                    for pat in data:
                        entries.append((str(pat), default_label))
            else:
                raise ValueError("Invalid JSON gazetteer structure.")
            return Gazetteer(entries)
        # Simple TXT/CSV file: one entry per line.
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pat = line.strip()
                if pat:
                    entries.append((pat, default_label))
        return Gazetteer(entries)

    def exact_spans(self, text: str) -> List[Span]:
        """
        Return spans that match gazetteer entries exactly.

        :param text: Source text.
        :type text: str
        :returns: Exact matches found within ``text``.
        :rtype: List[Span]
        """
        spans: List[Span] = []
        for label, kp in self.kp_by_label.items():
            for match, start, end in kp.extract_keywords(text, span_info=True):
                spans.append(Span(match, start, end, label, "gazetteer_exact", priority=100))
        return spans

    def fuzzy_spans(self, text: str, threshold: int = GAZ_FUZZY_THRESHOLD) -> List[Span]:
        """
        Return spans loosely matching gazetteer entries using fuzzy search.

        :param text: Source text.
        :type text: str
        :param threshold: Minimum fuzzy similarity score to accept a match.
        :type threshold: int
        :returns: Spans matched via fuzzy n-gram search.
        :rtype: List[Span]
        """
        spans: List[Span] = []
        words = text.split()
        offsets, pos = [], 0
        for w in words:
            offsets.append(pos)
            pos += len(w) + 1
        for label, vocab in self.entries_by_label.items():
            if not vocab:
                continue
            for n in range(1, MAX_NGRAM_WORDS + 1):
                for i in range(0, max(0, len(words) - n + 1)):
                    cand_words = words[i:i + n]
                    # Ignore n-grams made exclusively of stopwords or too short.
                    if all(w.lower() in STOPWORDS_FR for w in cand_words):
                        continue
                    cand = " ".join(cand_words)
                    if _alnum_len(cand) < 3:
                        continue
                    alnum_len = len(re.sub(r"[^A-Za-z0-9]", "", cand))
                    if alnum_len < 3:
                        continue
                    best = process.extractOne(cand, vocab, scorer=fuzz.WRatio)
                    if best and best[1] >= threshold:
                        start = offsets[i]
                        end = start + len(cand)
                        spans.append(Span(cand, start, end, label, "gazetteer_fuzzy", priority=60))
        return spans


def _detect_regex_spans(text: str) -> List[Span]:
    """
    Detect spans using regex patterns for emails, phone numbers, and titles.

    :param text: Source text.
    :type text: str
    :returns: Regex-derived spans prioritised above gazetteer matches.
    :rtype: List[Span]
    """
    spans: List[Span] = []
    # Priority order: EMAIL > TEL > TITLE+PERSON > gazetteer.
    for m in REGEX_EMAIL.finditer(text):
        spans.append(Span(m.group(), m.start(), m.end(), "EMAIL", "regex", 110))
    for m in REGEX_TEL.finditer(text):
        spans.append(Span(m.group(), m.start(), m.end(), "TEL", "regex", 105))
    for m in REGEX_TITLE_PERSON.finditer(text):
        a, b = _expand_to_word_boundaries(text, m.start(), m.end())
        spans.append(Span(text[a:b], a, b, "PER", "regex", 102))
    return spans



def _resolve_overlaps(spans: List[Span]) -> List[Span]:
    """
    Resolve overlapping spans by enforcing priority and length heuristics.

    :param spans: Candidate spans that may overlap.
    :type spans: List[Span]
    :returns: Ordered list without conflicting spans.
    :rtype: List[Span]
    """
    if not spans:
        return spans

    # Step 0: keep EMAIL/TEL spans dominant by removing nested spans entirely within them.
    emails = [s for s in spans if s.label == "EMAIL"]
    tels   = [s for s in spans if s.label == "TEL"]
    protected = emails + tels
    if protected:
        kept = []
        for s in spans:
            if any(p.start <= s.start and s.end <= p.end and s is not p for p in protected):
                continue  # remove sub-spans (e.g. PERSON inside an email local part)
            kept.append(s)
        spans = kept

    # Step 1: remove any remaining spans included in EMAIL/TEL spans.
    protected = [s for s in spans if s.label in ("EMAIL", "TEL")]
    if protected:
        kept = []
        for s in spans:
            if any(p.start <= s.start and s.end <= p.end and s is not p for p in protected):
                continue
            kept.append(s)
        spans = kept

    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start), -s.priority))
    out: List[Span] = []
    for s in spans:
        collide = None
        for i, t in enumerate(out):
            if not (s.end <= t.start or s.start >= t.end):
                collide = i; break
        if collide is None:
            out.append(s)
        else:
            t = out[collide]
            if (s.priority > t.priority) or (s.priority == t.priority and (s.end - s.start) > (t.end - t.start)):
                out[collide] = s
    return sorted(out, key=lambda s: s.start)

def _norm_key(s: str) -> str:
    """
    Produce a normalised key suitable for fuzzy comparisons.

    :param s: Input string.
    :type s: str
    :returns: Normalised comparison key.
    :rtype: str
    """
    s = unidecode(s)
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _group_aliases_generic(spans: List[Span], label: str, sim_threshold: int = 92) -> Dict[int, List[int]]:
    """
    Group spans of a given label (e.g. ORG) into similarity-based clusters.

    :param spans: Span collection.
    :type spans: List[Span]
    :param label: Label to cluster.
    :type label: str
    :param sim_threshold: Fuzzy similarity threshold.
    :type sim_threshold: int
    :returns: Mapping ``cluster_id -> [span_indices]``.
    :rtype: Dict[int, List[int]]
    """
    # Indices of spans matching the requested label.
    idxs = [i for i, s in enumerate(spans) if s.label == label]
    clusters: List[List[int]] = []
    for i in idxs:
        si = spans[i].text
        key_i = _norm_key(si)
        placed = False
        for cl in clusters:
            j = cl[0]
            sj = spans[j].text
            key_j = _norm_key(sj)

            # Matching rules: exact (normalised), all-caps acronyms, or fuzzy similarity.
            exact = (key_i == key_j)
            allcaps = re.fullmatch(r"[A-Z0-9&\-]{2,12}", si) and re.fullmatch(r"[A-Z0-9&\-]{2,12}", sj)
            sim = fuzz.WRatio(key_i, key_j)
            if exact or allcaps or sim >= sim_threshold:
                cl.append(i); placed = True; break
        if not placed:
            clusters.append([i])

    # Index clusters by order of first appearance in the text.
    clusters.sort(key=lambda ids: min(spans[k].start for k in ids))
    return {k+1: v for k, v in enumerate(clusters)}


def group_person_aliases(
    spans: List[Span],
    original_text: str,
    include_cat_as_person: bool = True,
) -> Dict[int, List[int]]:
    """
    Cluster personal entities based on textual similarity and email tokens.

    :param spans: Candidate spans (already deduplicated and ordered).
    :type spans: List[Span]
    :param original_text: Full text used to extract email hints.
    :type original_text: str
    :param include_cat_as_person: Treat ``CAT`` labels as potential persons.
    :type include_cat_as_person: bool
    :returns: Mapping ``cluster_id -> [span_indices]``.
    :rtype: Dict[int, List[int]]
    """
    email_sets = _extract_email_name_sets(original_text)

    def share_email_tokens(si: str, sj: str) -> bool:
        ti = _extended_name_tokens(si)
        tj = _extended_name_tokens(sj)
        for sset in email_sets:
            # Direct match (including nickname mappings).
            if (ti & sset) and (tj & sset):
                return True
            # Or approximate match against email tokens (e.g. "xav" ~ "xavier").
            for t in ti:
                if any(t == u or t.startswith(u) or u.startswith(t) or fuzz.WRatio(t, u) >= 88 for u in sset):
                    for v in tj:
                        if any(v == u or v.startswith(u) or u.startswith(v) or fuzz.WRatio(v, u) >= 88 for u in sset):
                            return True
        return False

    target_labels = ("PER", "CAT") if include_cat_as_person else ("PER",)
    idxs = [i for i, s in enumerate(spans) if s.label in target_labels]
    clusters: List[List[int]] = []
    for i in idxs:
        si = spans[i].text
        last_i, init_i = _last_and_initials(si)
        placed = False
        for cl in clusters:
            j = cl[0]; sj = spans[j].text
            last_j, init_j = _last_and_initials(sj)
            sim = fuzz.WRatio(_normalize_name(si), _normalize_name(sj))
            same_last = last_i and last_i == last_j
            same_init = init_i and init_i == init_j
            short_alias = min(len(_normalize_name(si)), len(_normalize_name(sj))) <= 4
            close_prefix = _normalize_name(si).startswith(_normalize_name(sj)) or _normalize_name(sj).startswith(_normalize_name(si))

            if share_email_tokens(si, sj) or \
               sim >= ALIAS_SIM_THRESHOLD or \
               (same_last and (sim >= 70 or same_init)) or \
               (short_alias and (sim >= 70 or close_prefix)):
                cl.append(i); placed = True; break
        if not placed:
            clusters.append([i])

    # Heuristic relabelling: single-word capitalised ORG similar to a PERSON cluster -> PER.
    for i, s in enumerate(spans):
        if s.label == "ORG" and _alnum_len(s.text) >= 4 and " " not in s.text:
            for ids in clusters:
                rep = spans[ids[0]].text
                if fuzz.WRatio(_normalize_name(s.text), _normalize_name(rep)) >= 88:
                    s.label = "PER"
                    ids.append(i)
                    break

    return {k+1: v for k,v in enumerate(clusters)}

@dataclass
class MapEntry:
    """Mapping entry describing anonymised tags and canonical mentions."""

    canonical: str
    mentions: List[str] = field(default_factory=list)
    type: str = "PERSON"

def _pick_canonical(mentions: Iterable[str]) -> str:
    """
    Select the canonical mention, favouring the longest variant.

    :param mentions: Iterable of surface forms.
    :type mentions: Iterable[str]
    :returns: Canonical mention string.
    :rtype: str
    """
    return max(mentions, key=lambda s: len(s))

def _apply_replacements(
    text: str,
    spans: List[Span],
    person_clusters: Dict[int, List[int]],
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Replace detected spans with anonymised tags while keeping spacing consistent.

    :param text: Original text.
    :type text: str
    :param spans: Final list of spans after conflict resolution.
    :type spans: List[Span]
    :param person_clusters: Mapping of PERSON clusters to span indices.
    :type person_clusters: Dict[int, List[int]]
    :returns: Tuple containing the anonymised text and the replacement mapping.
    :rtype: Tuple[str, Dict[str, Dict[str, Any]]]
    """
    repls, mapping = [], {}

    # 1) PERSON → <NOM_i>
    for cid, idxs in person_clusters.items():
        mentions = [spans[i].text for i in idxs]
        tag = f"<NOM_{cid}>"
        mapping[tag] = {"canonical": _pick_canonical(mentions), "mentions": sorted(set(mentions)), "type": "PERSON"}
        for i in idxs:
            s = spans[i]; repls.append((s.start, s.end, tag))

    used = set(i for ids in person_clusters.values() for i in ids)

    # 2) ORG / LIEU / CAT: cluster those labels as well.
    for lbl, tag_base, thr in (("ORG", "ORG", 92), ("LIEU", "LIEU", 94), ("CAT", "CAT", 95)):
        cl = _group_aliases_generic([spans[i] for i in range(len(spans))], lbl, sim_threshold=thr)
        # Convert cluster indices back to global span indices.
        clusters = {}
        label_idxs = [i for i, s in enumerate(spans) if s.label == lbl]
        for k, local_ids in cl.items():
            clusters[k] = [idx for idx in local_ids if idx in label_idxs]
        if not label_idxs:
            continue

        # Renumber clusters by global order for stability and dedicated per-label counters.
        counter = 0
        # Ensure every mention inside a cluster shares the same tag.
        cluster_values = [ids for ids in clusters.values() if ids]
        if not cluster_values:
            continue
        clusters_sorted = sorted(cluster_values, key=lambda ids: min(spans[i].start for i in ids))
        for idxs in clusters_sorted:
            counter += 1
            tag = f"<{tag_base}_{counter}>"
            mentions = [spans[i].text for i in idxs]
            mapping[tag] = {"canonical": _pick_canonical(mentions), "mentions": sorted(set(mentions)), "type": tag_base}
            for i in idxs:
                if i in used:  # avoid overriding PERSON already assigned (rare when label differs)
                    continue
                s = spans[i]; repls.append((s.start, s.end, tag))
                used.add(i)

    # 3) EMAIL / TEL (no clustering; these are exact and high priority).
    counters = {}
    for i, s in enumerate(spans):
        if i in used:
            continue
        if s.label in ("EMAIL", "TEL"):
            counters[s.label] = counters.get(s.label, 0) + 1
            tag = f"<{s.label}_{counters[s.label]}>"
            mapping[tag] = {"canonical": s.text, "mentions": [s.text], "type": s.label}
            repls.append((s.start, s.end, tag))
            used.add(i)

    # 4) Apply replacements while preserving surrounding spaces.
    repls.sort(key=lambda x: x[0], reverse=True)
    out = text
    for a, b, tag in repls:
        left_space  = " " if (a > 0 and not text[a-1].isspace()) else ""
        right_space = " " if (b < len(text) and not text[b:b+1].isspace()) else ""
        out = out[:a] + left_space + tag + right_space + out[b:]
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out, mapping


class Anonymizer:
    """Entity anonymiser orchestrating gazetteer, regex, and NER detectors."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: int | str | None = "auto") -> None:
        """
        Initialise the anonymiser and load the NER model.

        :param model_name: HuggingFace model name used for NER.
        :type model_name: str
        :param device: Device specifier (``"auto"``, ``"cpu"``, ``"cuda"``, or device index).
        :type device: int | str | None
        """
        if device == "auto":
            device = 0 if torch.cuda.is_available() else -1
        elif isinstance(device, str):
            device = 0 if device == "cuda" else -1
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner = hf_pipeline("ner", model=model, tokenizer=tokenizer, device=device)

    def anonymize(self, text: str,
                  catalog_path: Optional[str] = None,
                  catalog_label_default: str = "CAT",
                  catalog_fuzzy_threshold: int = GAZ_FUZZY_THRESHOLD,
                  catalog_as_person: bool = True) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """
        Anonymise input text and return the transformed text along with a mapping.

        :param text: Raw input text.
        :type text: str
        :param catalog_path: Optional gazetteer path to enrich detection.
        :type catalog_path: Optional[str]
        :param catalog_label_default: Default label applied to gazetteer entries.
        :type catalog_label_default: str
        :param catalog_fuzzy_threshold: Fuzzy threshold for gazetteer matching (0 disables fuzzy).
        :type catalog_fuzzy_threshold: int
        :param catalog_as_person: Treat gazetteer ``CAT`` entries as potential persons.
        :type catalog_as_person: bool
        :returns: Tuple of anonymised text and a mapping of tags to original mentions.
        :rtype: Tuple[str, Dict[str, Dict[str, Any]]]
        """
        gaz = Gazetteer.load(catalog_path, default_label=catalog_label_default) if catalog_path else None
        spans: List[Span] = []
        if gaz:
            spans += gaz.exact_spans(text)
            if catalog_fuzzy_threshold > 0:
                spans += gaz.fuzzy_spans(text, threshold=catalog_fuzzy_threshold)
        spans += _detect_regex_spans(text)
        spans += _detect_ner_spans_chunked(text, self.ner)

        prot = _protected_ranges(text)
        spans = _post_label_corrections(spans, text)
        spans = _drop_spans_in_protected(spans, prot)
        spans_final = _resolve_overlaps(spans)
        clusters = group_person_aliases(spans_final, original_text=text, include_cat_as_person=catalog_as_person)
        out, mapping = _apply_replacements(text, spans_final, clusters)  # ✅ keep using spans_final
        return out, mapping

    @staticmethod
    def deanonymize(text: str, mapping: Dict[str, Dict[str, Any]], restore: str = "canonical") -> str:
        """
        Restore anonymised placeholders using the provided mapping.

        :param text: Text containing anonymised tags.
        :type text: str
        :param mapping: Mapping produced by :meth:`anonymize`.
        :type mapping: Dict[str, Dict[str, Any]]
        :param restore: Restoration strategy, ``"canonical"`` or ``"first_mention"``.
        :type restore: str
        :returns: Text with placeholders replaced by original mentions.
        :rtype: str
        """
        items = list(mapping.items())
        for tag, info in sorted(items, key=lambda kv: -len(kv[0])):
            if restore == "first_mention" and info.get("mentions"):
                val = info["mentions"][0]
            else:
                val = info["canonical"]
            text = text.replace(tag, val)
        return text
