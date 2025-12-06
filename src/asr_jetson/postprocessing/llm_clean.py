"""LLM-based text clean-up utilities for French transcription outputs."""
from __future__ import annotations
import os
from pathlib import Path
import json
import re
import tempfile
import time
import requests
from typing import Iterator, Optional


def _prepare_lt_home() -> str:
    """
    Ensure ``LT_HOME`` points to a writable, persistent cache directory.

    The default /app/.cache path works inside our Docker image, but when the CLI
    runs directly on a developer machine that path is often read-only, causing
    ``language_tool_python`` to re-download LanguageTool into temporary folders
    every time. By selecting a user-specific cache location we only download the
    LanguageTool bundle once.
    """
    env_value = os.environ.get("LT_HOME")
    candidates = []

    if env_value:
        candidates.append(Path(env_value))
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            candidates.append(Path(xdg_cache) / "LanguageTool")
        candidates.append(Path.home() / ".cache" / "LanguageTool")
        candidates.append(Path.cwd() / ".cache" / "LanguageTool")
        candidates.append(Path(tempfile.gettempdir()) / "LanguageTool")

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        resolved = str(path)
        # language_tool_python reads LTP_PATH; keep LT_HOME for compatibility.
        os.environ.setdefault("LT_HOME", resolved)
        os.environ.setdefault("LTP_PATH", resolved)
        return str(path)

    raise RuntimeError("Unable to prepare a writable LT_HOME directory.")


# --- LanguageTool: persistent cache + lazy global instance ---
_LT_ENDPOINT = os.getenv("LT_ENDPOINT", "").strip() or None
_LT_DISABLED = os.getenv("DISABLE_LANGUAGETOOL", "").strip().lower() in {"1", "true", "yes", "on"}
_LT_INIT_DONE = False
_LT_INIT_ERROR: Optional[str] = None
LT_TOOL = None


def _ensure_language_tool():
    """
    Lazily instantiate LanguageTool, keeping the pipeline robust when Java/binaries
    are missing or crashy. Returns ``None`` when disabled or unavailable.
    """
    global _LT_INIT_DONE, _LT_INIT_ERROR, LT_TOOL
    if LT_TOOL is not None:
        return LT_TOOL
    if _LT_DISABLED or _LT_INIT_DONE:
        return None

    _LT_INIT_DONE = True
    try:
        cache_dir = _prepare_lt_home()
        # language_tool_python uses LTP_PATH; ensure it is set before import.
        os.environ.setdefault("LTP_PATH", cache_dir)
        import language_tool_python  # local import to avoid import-time side effects

        LT_TOOL = (
            language_tool_python.LanguageTool("fr", remote_server=_LT_ENDPOINT)
            if _LT_ENDPOINT
            else language_tool_python.LanguageTool("fr")
        )
        return LT_TOOL
    except Exception as _e:
        _LT_INIT_ERROR = str(_e)
        LT_TOOL = None
        print(f"[WARN] LanguageTool unavailable ({_e})")
        return None

os.environ["LLM_ENDPOINT"] = "http://tensorrt-llm:8000"
os.environ["LLM_MODEL"] = "gemma2:2b"

SYS_PROMPT = (
    "Tu es un correcteur automatique de transcription française. "
    "Règles: 1) Corrige orthographe, grammaire, accords, ponctuation française (espaces fines avant ; : ! ? facultatives, mais espace normal OK), "
    "2) Garde le sens AU PLUS PROCHE possible, 3) Ne SUPPRIME pas d'information sémantique, "
    "4) Ne réinvente rien, 5) Conserve la structure et les tags de locuteur 'SPEAKER_X:' "
    "6) Fusionne les micro-bégaiements évidents, 7) Ne change pas les numéros de SPEAKER."
    "8) Tu n’écris JAMAIS de commentaires ni d’analyses. "
    "9) Tu produis UNIQUEMENT le texte corrigé, sans aucun ajout ni retrait. "
    "10) Ton rôle est mécanique, comme un correcteur orthographique humain discipliné."
)

FEW_SHOT_EXAMPLES = """
EXEMPLE 1:
Entrée:
SPEAKER_1: bonjour je mappel jean je suis content detre ici
SPEAKER_2: enchanté moi cest marie

Sortie:
SPEAKER_1: Bonjour, je m'appelle Jean. Je suis content d'être ici.
SPEAKER_2: Enchanté, moi c'est Marie.

EXEMPLE 2:
Entrée:
SPEAKER_1: decleration des droit de lhomme et du citoyen
SPEAKER_2: oui cest un texte fondementale

Sortie:
SPEAKER_1: Déclaration des droits de l'homme et du citoyen.
SPEAKER_2: Oui, c'est un texte fondamental.
"""

USER_INSTR = (
    "IMPORTANT : Ce qui suit N'EST PAS une analyse, pas un résumé, pas une reformulation.\n"
    "Tu n'es PAS critique littéraire, tu es un correcteur automatique.\n"
    "TA SEULE MISSION : corriger les fautes d’orthographe, de grammaire et de ponctuation du texte donné, "
    "sans rien changer d’autre.\n"
    "Règles obligatoires :\n"
    "1️) Ne supprime RIEN.\n"
    "2️) Ne résume RIEN.\n"
    "3️) Ne commente RIEN.\n"
    "4️) Ne réécris PAS le style.\n"
    "5️) Ne traduis PAS.\n"
    "6️) Tu dois garder le même format exact que le texte d’entrée : mêmes lignes, mêmes préfixes 'SPEAKER_X:'.\n"
    "7️) Si tu sors autre chose que le texte corrigé, la sortie sera REJETÉE.\n"
    "Format attendu (exemple) :\n"
    "Entrée : SPEAKER_1: bonjour je suis alle a la plage\n"
    "Sortie : SPEAKER_1: Bonjour, je suis allé à la plage.\n"
    "Ne réponds QUE par la version corrigée, sans introduction, sans explication.\n"
    + FEW_SHOT_EXAMPLES +
    "MAINTENANT TON TOUR, voici le texte à corriger :\n"
)

def _basic_local_cleanup(text: str) -> str:
    """
    Perform minimal local clean-up when no LLM is available, keeping the pipeline safe.

    :param text: Raw transcription text.
    :type text: str
    :returns: Lightly cleaned text.
    :rtype: str
    """
    import re

    # 1. Basic whitespace reduction.
    s = re.sub(r"\s+", " ", text).strip()

    # 2. Punctuation spacing: remove preceding spaces, enforce trailing spaces.
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:!?])(?=\S)", r"\1 ", s)

    # 3. Normalise apostrophes.
    s = s.replace(" '", " ’").replace("' ", "’ ").replace("'", "’")

    # 4. Capitalise sentence starts.
    s = re.sub(r"(^|[.!?]\s+)(\w)", lambda m: m.group(1) + m.group(2).upper(), s)

    # 5. Place speaker tags on new lines with a single trailing space.
    s = re.sub(r"\s*(SPEAKER_\d+:)\s*", r"\n\1 ", s).strip()

    # 6. Remove duplicate newlines.
    s = re.sub(r"\n+", "\n", s)

    # 7. Ensure a trailing newline.
    if not s.endswith("\n"):
        s += "\n"

    return s

def _chunk_by_speakers(text: str, max_chars: int = 2200, max_lines: int = 80) -> Iterator[str]:
    """
    Split text into speaker-aligned chunks kept within context limits.

    :param text: Full transcription text.
    :type text: str
    :param max_chars: Maximum characters per chunk.
    :type max_chars: int
    :param max_lines: Maximum lines per chunk.
    :type max_lines: int
    :yields: Subsections of the transcription bounded by ``SPEAKER_X`` lines.
    :rtype: Iterator[str]
    """
    chunk, nchar, nline = [], 0, 0
    for line in text.splitlines():
        L = len(line) + 1
        if (nchar + L > max_chars) or (nline + 1 > max_lines):
            if chunk:
                yield "\n".join(chunk)
            chunk, nchar, nline = [], 0, 0
        chunk.append(line); nchar += L; nline += 1
    if chunk:
        yield "\n".join(chunk)

def _dedupe_runs(text: str, k_lines: int = 3, max_repeats: int = 2) -> str:
    """
    Collapse consecutive repetitions of ``k_lines`` to avoid LLM feedback loops.

    :param text: Input text potentially containing repeated blocks.
    :type text: str
    :param k_lines: Window size to compare for repetition.
    :type k_lines: int
    :param max_repeats: Maximum allowed repetitions before trimming.
    :type max_repeats: int
    :returns: Text with redundant repeated blocks removed.
    :rtype: str
    """
    lines, out, rep = text.splitlines(), [], 0
    for ln in lines:
        out.append(ln)
        if len(out) >= 2*k_lines and out[-k_lines:] == out[-2*k_lines:-k_lines]:
            rep += 1
            if rep >= max_repeats:
                out = out[:-k_lines]  # trim the repeated series
                rep = 0
        else:
            rep = 0
    return "\n".join(out)


def _call_openai_compatible(
    endpoint: str,
    api_key: Optional[str],
    model: str,
    text: str,
    timeout_s: int = 120,
) -> Optional[str]:
    """
    Invoke an OpenAI-compatible chat completion endpoint.

    :param endpoint: Base URL of the OpenAI-compatible server.
    :type endpoint: str
    :param api_key: Optional bearer token.
    :type api_key: Optional[str]
    :param model: Model identifier to request.
    :type model: str
    :param text: Text block to correct.
    :type text: str
    :param timeout_s: Request timeout in seconds.
    :type timeout_s: int
    :returns: Corrected text or ``None`` on failure.
    :rtype: Optional[str]
    """
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Reduce the user prompt if the block is long (remove few-shot examples).
    user_msg = USER_INSTR + "\n\n" + text
    if len(text.split()) > 1200:
        user_msg = USER_INSTR.replace(FEW_SHOT_EXAMPLES, "") + "\n\n" + text

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "top_p": 0.92,                       # small exploration margin without drifting
        "max_tokens": min(2048, max(512, len(text.split()) * 2)),  # hard upper bound
        "stop": ["\n\nSPEAKER_", "\nSPEAKER_"],  # avoid bleeding into subsequent blocks
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if r.status_code == 200:
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    return None


# def _call_ollama(model: str, text: str, timeout_s: int = 120) -> Optional[str]:
#     """
#     Appelle un serveur Ollama local (http://localhost:11434) si dispo.
#     """
#     url = "http://localhost:11434/api/chat"
#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": SYS_PROMPT},
#             {"role": "user", "content": USER_INSTR + "\n\n" + text},
#         ],
#         "options": {"temperature": 0.1},
#         "stream": False,
#     }
#     r = requests.post(url, json=payload, timeout=timeout_s)
#     if r.status_code == 200:
#         return r.json().get("message", {}).get("content", "").strip()
#     return None

def _call_ollama(model: str, text: str, timeout_s: int = 120) -> Optional[str]:
    """
    Call the Ollama ``/api/chat`` endpoint with settings tuned for Gemma 2.

    - Avoid seeding with an ``assistant`` message at the end (breaks some templates).
    - Increase ``num_ctx`` to prevent truncating long transcripts.
    - Remove few-shot examples when the input is already long.

    :param model: Ollama model identifier.
    :type model: str
    :param text: Text block to correct.
    :type text: str
    :param timeout_s: Request timeout in seconds.
    :type timeout_s: int
    :returns: Corrected text or ``None`` if the request fails validation.
    :rtype: Optional[str]
    """
    import re
    url = "http://localhost:11434/api/chat"

    # Remove examples when the text is long to preserve context.
    user_msg = USER_INSTR + "\n\n" + text
    if len(text.split()) > 1200:  # empirical threshold
        user_msg = USER_INSTR.replace(FEW_SHOT_EXAMPLES, "") + "\n\n" + text

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS_PROMPT.replace("JA%AIS", "JAMAIS")},
            {"role": "user",   "content": user_msg},
            # Important: do not append an assistant message here.
        ],
        "options": {
            "temperature": 0.0,
            "top_p": 0.92,
            "repeat_penalty": 1.25,
            "num_predict": max(512, len(text.split()) * 2),
            "num_ctx": 8192,  # Critical: increase context window if the model supports it.
            "stop": ["\n\nSPEAKER_", "\nSPEAKER_"]
        },
        "stream": False,
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code == 200:
            content = r.json().get("message", {}).get("content", "").strip()
            # Sanity check: if the model did not ingest the text it may respond with prompts such as
            # "Commencez..." or "J'attends...".
            lower = content.lower()
            if ("commencez" in lower or "j'attends votre transcription" in lower
                or ("corriger" in lower and len(content.split()) < 12)):
                # Retry using the strict prefix generation mode.
                return _ollama_generate_prefix(model, text, timeout_s)
            # Validate the output (speaker tags, length, etc.).
            if _validate_ollama_output(text, content):
                return content
        else:
            print(f"⚠️ Ollama HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
    return None

def _ollama_generate_prefix(model: str, text: str, timeout_s: int = 120) -> Optional[str]:
    """
    Fallback using ``/api/generate`` with a strict ``SPEAKER_X:`` prefix to enforce format.

    :param model: Ollama model identifier.
    :type model: str
    :param text: Text block to correct.
    :type text: str
    :param timeout_s: Request timeout in seconds.
    :type timeout_s: int
    :returns: Corrected text or ``None`` when output validation fails.
    :rtype: Optional[str]
    """
    import json
    import re
    import requests
    url = "http://localhost:11434/api/generate"
    first = re.search(r'(SPEAKER_\d+:)', text)
    prefix = (first.group(1) + " ") if first else "SPEAKER_1: "
    prompt = (
        SYS_PROMPT.replace("JA%AIS", "JAMAIS")
        + "\n\n"
        + USER_INSTR.replace(FEW_SHOT_EXAMPLES, "")
        + "\n\n"
        + "Réponds en commençant exactement par: " + prefix + "\n\n"
        + text
        + "\n\n"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": 0.0, "num_predict": max(512, len(text.split()) * 2), "num_ctx": 8192},
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    if r.status_code == 200:
        out = r.json().get("response", "").strip()
        # Guarantee the prefix is present.
        if not out.startswith(prefix):
            out = prefix + out
        return out if _validate_ollama_output(text, out) else None
    return None


def _validate_ollama_output(original: str, corrected: str) -> bool:
    """
    Ensure the Ollama output is a correction rather than an analysis.

    :param original: Original transcription.
    :type original: str
    :param corrected: Candidate corrected output.
    :type corrected: str
    :returns: ``True`` if the corrected text respects constraints, ``False`` otherwise.
    :rtype: bool
    """

    # 1. Check speaker tags.
    orig_speakers = set(re.findall(r'SPEAKER_\d+:', original))
    corr_speakers = set(re.findall(r'SPEAKER_\d+:', corrected))
    if orig_speakers != corr_speakers:
        print(f"⚠️ Speaker tags mismatch: {orig_speakers} vs {corr_speakers}")
        return False

    # 2. Check length stays within ±50%.
    orig_words = len(original.split())
    corr_words = len(corrected.split())
    if corr_words < orig_words * 0.5 or corr_words > orig_words * 1.5:
        print(f"⚠️ Suspicious length: {orig_words} words → {corr_words} words")
        return False

    # 3. Detect analysis-style introductions.
    forbidden_start = ["voici", "cette", "le texte", "analyse", "résumé", "thème"]
    first_line = corrected.split('\n')[0].lower()
    if any(first_line.startswith(word) for word in forbidden_start):
        print(f"⚠️ Analysis-style output detected: {first_line[:100]}")
        return False

    # 4. Reject numbered bullet structures.
    if re.search(r'^\d+\.', corrected, re.MULTILINE) or corrected.count('-') > orig_words * 0.1:
        print("⚠️ Analysis structure detected")
        return False

    return True

def _validate_structure(original: str, corrected: str) -> bool:
    """
    Validate that the structure and speaker tags remain unchanged.

    :param original: Original transcription.
    :type original: str
    :param corrected: Candidate corrected output.
    :type corrected: str
    :returns: ``True`` if the structure matches, ``False`` otherwise.
    :rtype: bool
    """
    import re
    orig_tags = re.findall(r'^(\s*SPEAKER_\d+:\s*)', original, flags=re.MULTILINE)
    corr_tags = re.findall(r'^(\s*SPEAKER_\d+:\s*)', corrected, flags=re.MULTILINE)
    return orig_tags == corr_tags and set(re.findall(r'SPEAKER_\d+:', original)) == set(re.findall(r'SPEAKER_\d+:', corrected))

def _force_sentence_caps(text: str) -> str:
    """
    Capitalise sentence starts without modifying speaker tags.

    :param text: Text requiring sentence-case adjustments.
    :type text: str
    :returns: Text with adjusted casing.
    :rtype: str
    """
    import re
    def fix_line(line: str) -> str:
        line = re.sub(r'^([a-zà-ÿ])', lambda m: m.group(1).upper(), line)
        line = re.sub(r'([\.!?…]\s+)([a-zà-ÿ])', lambda m: m.group(1)+m.group(2).upper(), line)
        return line
    out = []
    for ln in text.splitlines():
        m = re.match(r'^(\s*SPEAKER_\d+:\s*)(.*)$', ln)
        if not m:
            out.append(ln); continue
        out.append(m.group(1) + fix_line(m.group(2)))
    return "\n".join(out) + "\n"

def _lowercase_connectives_after_comma(text: str) -> str:
    """
    Lowercase select connectives appearing after a comma mid-sentence (e.g. ", Et" → ", et").

    The function preserves ``SPEAKER_X`` prefixes and only adjusts listed words.

    :param text: Text requiring connective adjustments.
    :type text: str
    :returns: Text with normalised connective casing.
    :rtype: str
    """
    import re
    CONNECTEURS = {
        "Et", "Mais", "Car", "Comme", "Or", "Donc", "Puis",
        "Cependant", "Toutefois", "Néanmoins", "Alors", "Ainsi"
    }
    def fix_line(line: str) -> str:
        # ',  Mot' -> ', mot' si Mot ∈ CONNECTEURS
        def repl(m):
            word = m.group(2)
            return m.group(1) + word.lower()
        pattern = r'([,;:]\s+)(%s)\b' % "|".join(sorted(CONNECTEURS))
        return re.sub(pattern, repl, line)
    out = []
    for ln in text.splitlines():
        m = re.match(r'^(\s*SPEAKER_\d+:\s*)(.*)$', ln)
        if not m:
            out.append(ln); continue
        prefix, content = m.groups()
        out.append(prefix + fix_line(content))
    return "\n".join(out) + "\n"

def _fix_caps_with_languagetool(text: str) -> Optional[str]:
    """
    Apply only casing corrections via LanguageTool, line by line, preserving ``SPEAKER_X`` tags.

    :param text: Text to adjust.
    :type text: str
    :returns: Corrected text or ``None`` when LanguageTool is unavailable or invalidates structure.
    :rtype: Optional[str]
    """
    tool = _ensure_language_tool()
    if tool is None:
        return None
    import re
    out_lines = []
    for line in text.splitlines():
        m = re.match(r'^(\s*SPEAKER_\d+:\s*)(.*)$', line)
        if not m:
            out_lines.append(line); continue
        prefix, content = m.groups()
        matches = tool.check(content)
        edits = []
        for mt in matches:
            rid = (mt.ruleId or "").upper()
            cat = (getattr(mt, "category", None) or "").upper()
            if ("UPPER" in rid) or ("CAPITAL" in rid) or ("CASING" in cat):
                if mt.replacements:
                    edits.append((mt.offset, mt.errorLength, mt.replacements[0]))
        edits.sort(key=lambda x: x[0], reverse=True)
        buf = content
        for off, ln, repl in edits:
            buf = buf[:off] + repl + buf[off+ln:]
        out_lines.append(prefix + buf)
    corrected = "\n".join(out_lines).rstrip() + "\n"
    return corrected if _validate_structure(text, corrected) else None


def _clean_with_languagetool(text: str) -> Optional[str]:
    """
    Safe LanguageTool run applying only casing and punctuation/spacing fixes.

    Lexical replacements (e.g. "semble" → "paraît") are intentionally avoided.

    :param text: Text to adjust.
    :type text: str
    :returns: Corrected text or ``None`` when unavailable or rejected.
    :rtype: Optional[str]
    """
    try:
        tool = _ensure_language_tool()
        if tool is None:
            return None
        import re, unicodedata

        def is_punct_or_space_change(src: str, dst: str) -> bool:
            # True if difference only concerns punctuation or spaces.
            def keep_letters(s):  # remove punctuation + spaces
                return "".join(ch for ch in s if ch.isalpha() or ch.isdigit())
            return keep_letters(src).lower() == keep_letters(dst).lower()

        out_lines = []
        for line in text.splitlines():
            m = re.match(r'^(\s*SPEAKER_\d+:\s*)(.*)$', line)
            if not m:
                out_lines.append(line); continue

            prefix, content = m.groups()
            matches = tool.check(content)
            edits = []
            for mt in matches:
                rid = (mt.ruleId or "").upper()
                cat = (getattr(mt, "category", None) or "").upper()
                repl = mt.replacements[0] if mt.replacements else None
                if not repl:
                    continue
                # Allow when casing rules trigger or the difference is limited to punctuation/spaces.
                if ("UPPER" in rid) or ("CAPITAL" in rid) or ("CASING" in cat) or is_punct_or_space_change(content[mt.offset:mt.offset+mt.errorLength], repl):
                    edits.append((mt.offset, mt.errorLength, repl))

            edits.sort(key=lambda x: x[0], reverse=True)
            buf = content
            for off, ln, repl in edits:
                buf = buf[:off] + repl + buf[off+ln:]
            out_lines.append(prefix + buf)

        corrected_text = "\n".join(out_lines).rstrip() + "\n"
        return corrected_text if _validate_structure(text, corrected_text) else None
    except Exception as e:
        print(f"⚠️ LanguageTool unavailable: {e}")
        return None


def clean_text_with_llm(
    input_txt: Path,
    output_txt: Path,
    default_model: str = "gemma2:2b",
    timeout_s: int = 240,
) -> Path:
    """
    Clean transcription text using available LLM backends with safe fallbacks.

    :param input_txt: Path to the input text file.
    :type input_txt: Path
    :param output_txt: Destination path for the cleaned text.
    :type output_txt: Path
    :param default_model: Default Ollama model identifier.
    :type default_model: str
    :param timeout_s: Timeout for LLM requests in seconds.
    :type timeout_s: int
    :returns: Path to the cleaned text file.
    :rtype: Path
    """
    input_txt = Path(input_txt)
    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    raw = input_txt.read_text(encoding="utf-8")

    # Start from the raw text.
    current = raw

    # (1) OpenAI-compatible backend (if available).
    endpoint = os.getenv("LLM_ENDPOINT", "").strip()
    if endpoint:
        model = os.getenv("LLM_MODEL", default_model).strip() or default_model
        api_key = os.getenv("LLM_API_KEY", "").strip() or None
        try:
            pieces = []
            ok = True
            for blk in _chunk_by_speakers(raw, max_chars=2200, max_lines=80):
                out = _call_openai_compatible(endpoint, api_key, model, blk)
                if not out or not _validate_ollama_output(blk, out):
                    ok = False
                    break
                out = _dedupe_runs(out, k_lines=3, max_repeats=1)
                pieces.append(out.rstrip("\n"))
            if ok and pieces:
                merged = "\n".join(pieces).rstrip() + "\n"
                if _validate_structure(raw, merged):
                    current = merged
                    print("✅ LLM (OpenAI-compatible, chunked)")
                else:
                    print("⚠️ Invalid structure after reconciling LLM output — ignoring result.")
            else:
                print("⚠️ OpenAI-compatible LLM failed on at least one block — ignoring output.")
        except Exception as e:
            print(f"⚠️ OpenAI-compatible error: {e}")

    # (2) Ollama (only if the previous LLM produced nothing usable).
    if current is raw:
        use_ollama = (os.getenv("USE_OLLAMA", "1") == "1")
        if use_ollama:
            try:
                pieces = []
                ok = True
                for blk in _chunk_by_speakers(raw, max_chars=2200, max_lines=80):
                    out = _call_ollama(default_model, blk, timeout_s=timeout_s)
                    if not out or not _validate_ollama_output(blk, out):
                        ok = False
                        break
                    out = _dedupe_runs(out, k_lines=3, max_repeats=1)
                    pieces.append(out.rstrip("\n"))
                if ok and pieces:
                    merged = "\n".join(pieces).rstrip() + "\n"
                    if _validate_structure(raw, merged):
                        current = merged
                        print("✅ LLM (Ollama, chunked)")
                    else:
                        print("⚠️ Invalid structure after reconciling Ollama output — ignoring result.")
                else:
                    print("⚠️ Ollama failed on at least one block — ignoring output.")
            except Exception as e:
                print(f"⚠️ Ollama error: {e}")

        # --- From here: post-processing always applied on ``current`` ---
            # (3) Caps-first: regex + LanguageTool casing-only pass.
            current_caps = _force_sentence_caps(current)
            lt_caps = _fix_caps_with_languagetool(current_caps)
            if lt_caps and _validate_structure(current, lt_caps):
                current = lt_caps
                print("✅ Caps fix (regex + LT)")

            # (3bis) Lowercase connectives after commas (", Et" -> ", et", etc.).
            current = _lowercase_connectives_after_comma(current)

            # (4) LanguageTool safe pass (casing + punctuation/spacing only).
            lt_full = _clean_with_languagetool(current)
            if lt_full and _validate_structure(current, lt_full):
                current = lt_full
                print("✅ LanguageTool (casing + punctuation)")

            # (5) Minimal regex clean-up (soft normalisation).
            current = _basic_local_cleanup(current)

    # Final write-out.
    output_txt.write_text(current if current.endswith("\n") else current + "\n", encoding="utf-8")
    print("✅ Final post-processing complete")
    return output_txt
