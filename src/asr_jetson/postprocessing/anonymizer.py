from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable
from unidecode import unidecode
from flashtext import KeywordProcessor
from rapidfuzz import fuzz, process
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline

# --- Configs ---
DEFAULT_MODEL = "cmarkea/distilcamembert-base-ner"
GAZ_FUZZY_THRESHOLD = 90
ALIAS_SIM_THRESHOLD = 82  # ↓ un peu pour regrouper + robuste (Xav <-> Xavier)
MAX_NGRAM_WORDS = 5
ENTITY_CANON = {"PER":"NOM","ORG":"ORG","LOC":"LIEU","MISC":"CAT"}

# Stopwords FR pour bloquer les faux positifs fuzzy
STOPWORDS_FR = {
    "a","à","au","aux","chez","de","des","du","et","en","la","le","les","un","une","ce","cet","cette",
    "sur","dans","par","pour","plus","moins","ou","où","avec","sans","se","ses","son","sa","leurs","leur"
}

REGEX_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
REGEX_TEL   = re.compile(r"(?<!\w)(?:\+?\d[\s\-.]?){7,}(?!\w)")
REGEX_TITLE_PERSON = re.compile(
    r"\b(?:M|Mr|Monsieur|Mme|Madame|Mlle|Mademoiselle)\.?\s+[A-ZÉÈÀÂÎÔÙÜÇ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
)

# NEW: nicknames simples (ajoute ce que tu veux)
NICKNAME_MAP = {
    "xav": "xavier",
    "alex": "alexandre",
    "ben": "benjamin",
}

# NEW: "titre + nom" (gère M/Mr/Monsieur/Mme/Madame/Mlle/Mademoiselle, point optionnel)
REGEX_TITLE_PERSON = re.compile(
    r"\b(?:M|Mr|Monsieur|Mme|Madame|Mlle|Mademoiselle)\.?\s+"
    r"[A-ZÉÈÀÂÎÔÙÜÇ][A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
)

_WORD_CHARS = r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]"

def _expand_to_word_boundaries(text: str, start: int, end: int):
    a, b = start, end
    while a > 0 and re.match(_WORD_CHARS, text[a-1]):
        a -= 1
    while b < len(text) and re.match(_WORD_CHARS, text[b:b+1]):
        b += 1
    return a, b

def _alnum_len(s: str) -> int:
    return len(re.sub(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ]", "", s))

# Diminutifs FR (ajoute ce que tu veux au fil de l’eau)
NICK_EQUIV = {
    "xav": "xavier",
    "alex": "alexandre",
    "fred": "frederic",
    "nico": "nicolas",
    "ben": "benjamin",
}

def _extract_email_name_sets(text: str) -> list[set[str]]:
    sets = []
    for m in REGEX_EMAIL.finditer(text):
        local = m.group().split("@", 1)[0]
        toks = [unidecode(t).lower() for t in re.split(r"[._\-+]", local) if t]
        if toks:
            sets.append(set(toks))
    return sets

def _extended_name_tokens(s: str) -> set[str]:
    """Tokens normalisés + variantes via diminutifs/prefix pour mieux croiser avec les emails."""
    toks = [t for t in re.split(r"[\s\-]+", _normalize_name(s)) if t]
    out = set(toks)
    for t in list(out):
        if t in NICK_EQUIV:
            out.add(NICK_EQUIV[t])
    return out


def _nickname_normalize(s: str) -> str:
    base = re.sub(r"[^a-z]", "", _normalize_name(s))
    return NICKNAME_MAP.get(base, base)

def _post_label_corrections(spans: List[Span]) -> List[Span]:
    """
    CamemBERT peut taguer par erreur en ORG des tokens Person-like.
    Règles:
      - un seul token Capitalisé sans suffixe org → PER
      - proche (fuzzy) d'une autre mention PER → PER
    """
    # collecter mentions PER pour rapprochement
    per_texts = [s.text for s in spans if s.label == "PER"]
    corrected = []
    for s in spans:
        if s.label == "ORG":
            tok = s.text.strip()
            # un seul mot capitalisé (ex: 'Axavier') et pas un mot-clé d'entreprise
            if re.match(r"^[A-ZÉÈÀÂÎÔÙÜÇ][a-zà-öø-ÿ'’-]+$", tok) and not re.search(r"(SA|SARL|SAS|Inc|LLC|Ltd)\b", tok):
                # si proche d'une PER détectée
                if per_texts:
                    best = process.extractOne(tok, per_texts, scorer=fuzz.WRatio)
                    if best and best[1] >= 80:
                        s = Span(s.text, s.start, s.end, "PER", s.source, s.priority)
                else:
                    # par défaut reclasser en PER
                    s = Span(s.text, s.start, s.end, "PER", s.source, s.priority)
        corrected.append(s)
    return corrected

def _strip_titles(s: str) -> str:
    return re.sub(r"\b(mr|mme|mlle|m|m\.|mme\.|mlle\.)\b", "", s, flags=re.I)

def _normalize_name(s: str) -> str:
    s = unidecode(s)
    s = _strip_titles(s)
    s = re.sub(r"[^\w\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _last_and_initials(s: str) -> Tuple[str, str]:
    toks = [t for t in re.split(r"[\s\-]+", _normalize_name(s)) if t]
    if not toks:
        return "", ""
    last = toks[-1]
    initials = "".join(t[0] for t in toks)
    return last, initials

@dataclass
class Span:
    text: str
    start: int
    end: int
    label: str     # PER/ORG/LOC/EMAIL/TEL/CAT
    source: str    # ner/gazetteer_exact/gazetteer_fuzzy/regex
    priority: int

class Gazetteer:
    def __init__(self, entries: List[Tuple[str, str]]):
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
                raise ValueError("Catalogue JSON invalide.")
            return Gazetteer(entries)
        # txt/csv simple
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pat = line.strip()
                if pat:
                    entries.append((pat, default_label))
        return Gazetteer(entries)

    def exact_spans(self, text: str) -> List[Span]:
        spans: List[Span] = []
        for label, kp in self.kp_by_label.items():
            for match, start, end in kp.extract_keywords(text, span_info=True):
                spans.append(Span(match, start, end, label, "gazetteer_exact", priority=100))
        return spans

    def fuzzy_spans(self, text: str, threshold: int = GAZ_FUZZY_THRESHOLD) -> List[Span]:
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
                    # Ignore n-grammes constitués uniquement de stopwords ou trop courts
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
    spans: List[Span] = []
    # EMAIL > TEL > TITLE+PERSON > gazetteer
    # Priorités hautes pour coiffer tout le reste
    for m in REGEX_EMAIL.finditer(text):
        spans.append(Span(m.group(), m.start(), m.end(), "EMAIL", "regex", 110))
    for m in REGEX_TEL.finditer(text):
        spans.append(Span(m.group(), m.start(), m.end(), "TEL", "regex", 105))
    for m in REGEX_TITLE_PERSON.finditer(text):
        a, b = _expand_to_word_boundaries(text, m.start(), m.end())
        spans.append(Span(text[a:b], a, b, "PER", "regex", 102))
    return spans



def _detect_ner_spans(text: str, ner_pipe) -> List[Span]:
    spans: List[Span] = []
    for e in ner_pipe(text, aggregation_strategy="simple"):
        raw_lbl = e.get("entity_group") or e.get("entity", "")
        lbl = ENTITY_CANON.get(raw_lbl, raw_lbl)
        if lbl not in ("NOM", "ORG", "LIEU", "CAT"):
            continue
        a, b = _expand_to_word_boundaries(text, int(e["start"]), int(e["end"]))
        mention = text[a:b]
        if lbl == "CAT" and _alnum_len(mention) < 3:
            continue  # anti faux-positifs 'ce', 'a', ...
        spans.append(Span(mention, a, b, "PER" if lbl == "NOM" else lbl, "ner", 50))
    return spans

def _resolve_overlaps(spans: List[Span]) -> List[Span]:
    if not spans:
        return spans

    # 0) Protection : si un span EMAIL/TEL existe, on supprime tout span inclus dedans
    emails = [s for s in spans if s.label == "EMAIL"]
    tels   = [s for s in spans if s.label == "TEL"]
    protected = emails + tels
    if protected:
        kept = []
        for s in spans:
            if any(p.start <= s.start and s.end <= p.end and s is not p for p in protected):
                continue  # élimine sous-spans (ex: PERSON dans le local-part d'un email)
            kept.append(s)
        spans = kept

    # 1) Résolution des chevauchements restante (priorité + longueur)
    # supprime tout sous-span inclus dans un EMAIL/TEL
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

def group_person_aliases(spans: List[Span], original_text: str, include_cat_as_person: bool = True) -> Dict[int, List[int]]:
    email_sets = _extract_email_name_sets(original_text)

    def share_email_tokens(si: str, sj: str) -> bool:
        ti = _extended_name_tokens(si)
        tj = _extended_name_tokens(sj)
        for sset in email_sets:
            # match direct (incluant diminutifs mappés)…
            if (ti & sset) and (tj & sset):
                return True
            # …ou match "proche" (prefix/similarité) contre les tokens d'email
            # utile pour cas courts: "xav" ~ "xavier"
            for t in ti:
                if any(t == u or t.startswith(u) or u.startswith(t) or fuzz.WRatio(t, u) >= 88 for u in sset):
                    for v in tj:
                        if any(v == u or v.startswith(u) or u.startswith(v) or fuzz.WRatio(v, u) >= 88 for u in sset):
                            return True
        return False

    idxs = [i for i,s in enumerate(spans) if s.label in ("PER","CAT")]
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

    # Reclassement heuristique : ORG mono-mot capitalisé très proche d’un cluster PERSON → PERSON
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
    canonical: str
    mentions: List[str] = field(default_factory=list)
    type: str = "PERSON"

def _pick_canonical(mentions: Iterable[str]) -> str:
    return max(mentions, key=lambda s: len(s))

def _apply_replacements(text: str, spans: List[Span], clusters: Dict[int, List[int]]):
    repls, mapping = [], {}
    # PERSON clusters → <NOM_i>
    for cid, idxs in clusters.items():
        mentions = [spans[i].text for i in idxs]
        tag = f"<NOM_{cid}>"
        mapping[tag] = {"canonical": _pick_canonical(mentions), "mentions": sorted(set(mentions)), "type": "PERSON"}
        for i in idxs:
            s = spans[i]; repls.append((s.start, s.end, tag))
    # autres
    used = set(i for ids in clusters.values() for i in ids)
    counters: Dict[str,int] = {}
    for i,s in enumerate(spans):
        if i in used: continue
        label_out = {"PER":"NOM","ORG":"ORG","LIEU":"LIEU","EMAIL":"EMAIL","TEL":"TEL","CAT":"CAT"}.get(s.label, "CAT")
        if label_out == "NOM":
            label_out = "CAT"
        counters[label_out] = counters.get(label_out, 0) + 1
        tag = f"<{label_out}_{counters[label_out]}>"
        mapping[tag] = {"canonical": s.text, "mentions": [s.text], "type": label_out if label_out!="NOM" else "PERSON"}
        repls.append((s.start, s.end, tag))
    repls.sort(key=lambda x: x[0], reverse=True)
    out = text
    for a, b, tag in repls:
        left_space = " " if (a > 0 and not text[a - 1].isspace()) else ""
        right_space = " " if (b < len(text) and not text[b:b + 1].isspace()) else ""
        out = out[:a] + left_space + tag + right_space + out[b:]
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out, mapping

class Anonymizer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[int|str] = "auto"):
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
                  catalog_as_person: bool = True) -> Tuple[str, Dict[str, dict]]:
        gaz = Gazetteer.load(catalog_path, default_label=catalog_label_default) if catalog_path else None
        spans: List[Span] = []
        if gaz:
            spans += gaz.exact_spans(text)
            if catalog_fuzzy_threshold > 0:
                spans += gaz.fuzzy_spans(text, threshold=catalog_fuzzy_threshold)
        spans += _detect_regex_spans(text)
        spans += _detect_ner_spans(text, self.ner)
        spans = _post_label_corrections(spans)
        spans_final = _resolve_overlaps(spans)
        clusters = group_person_aliases(spans_final, original_text=text, include_cat_as_person=catalog_as_person)
        out, mapping = _apply_replacements(text, spans_final, clusters)  # ✅ utiliser spans_final
        return out, mapping

    @staticmethod
    def deanonymize(text: str, mapping: Dict[str, dict], restore: str = "canonical") -> str:
        """
        restore: "canonical" | "first_mention"
        """
        items = list(mapping.items())
        for tag, info in sorted(items, key=lambda kv: -len(kv[0])):
            if restore == "first_mention" and info.get("mentions"):
                val = info["mentions"][0]
            else:
                val = info["canonical"]
            text = text.replace(tag, val)
        return text
