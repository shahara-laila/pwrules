"""Hashcat rule inference from (base_word, password) pairs.

Algorithm
---------
Given a cleaned password P and a reference wordlist, this module:

1. Selects a base word B by:
   - Stripping leading / trailing non-alphabetic chars from P.
   - Reverse-mapping known leet substitutions.
   - Lowercasing and looking up the result in the wordlist.

2. Infers a Hashcat rule R that transforms B → P by detecting (in order):
   - Duplication (d)
   - Reversal (r)
   - Case operation (c / u / t / l / none)
   - Leet substitutions (sXY for each substituted char)
   - Prepended prefix (^X for each char, in reverse order)
   - Appended suffix ($X for each char)

3. Validates: applies the inferred rule to B and asserts apply_rule(B, R) == P.
   Triples that fail validation are discarded.

The rule applier is implemented in ``applier.py`` (pure Python). A separate
``parity_check`` function cross-validates a sample against real Hashcat stdout.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import os
from typing import Dict, FrozenSet, List, Optional, Tuple

from pwrules.ruleextract.applier import apply_rule

# ---------------------------------------------------------------------------
# Leet-speak map (canonical, lowercase source chars).
# The map is fixed — changes here would alter the extracted rule distribution.
# ---------------------------------------------------------------------------

DEFAULT_LEET_MAP: Dict[str, str] = {
    "a": "@",
    "e": "3",
    "o": "0",
    "i": "1",
    "s": "$",
    "l": "1",   # Note: i and l both → '1'; reverse prefers 'i'.
    "t": "+",
    "g": "9",
    "b": "8",
    "z": "2",
}

# Reverse leet: target char → source char (prefer the most common mapping).
_REVERSE_LEET: Dict[str, str] = {
    "@": "a",
    "3": "e",
    "0": "o",
    "1": "i",   # prefer 'i' over 'l' when reversing '1'
    "$": "s",
    "+": "t",
    "9": "g",
    "8": "b",
    "2": "z",
}


# ---------------------------------------------------------------------------
# Base-word selection
# ---------------------------------------------------------------------------

def _strip_outer_nonalpha(password: str) -> Tuple[str, str, str]:
    """Return (prefix, core, suffix) where core contains only the run of
    chars that starts and ends with an alphabetic character.

    Non-alphabetic chars embedded within the alphabetic run are kept (they
    may be leet substitutions like '@' for 'a').
    """
    if not password:
        return "", "", ""

    # Find the first and last alphabetic position.
    first_alpha = next((i for i, c in enumerate(password) if c.isalpha()), None)
    last_alpha = next(
        (i for i, c in enumerate(reversed(password)) if c.isalpha()), None
    )

    if first_alpha is None:
        # No alphabetic characters at all.
        return password, "", ""

    end = len(password) - last_alpha  # exclusive index of last alpha + 1
    prefix = password[:first_alpha]
    suffix = password[end:]
    core = password[first_alpha:end]
    return prefix, core, suffix


def _reverse_leet(text: str, rev_map: Dict[str, str] = _REVERSE_LEET) -> str:
    """Replace leet chars with their alphabetic equivalents."""
    return "".join(rev_map.get(c, c) for c in text)


def select_base(
    password: str,
    wordlist: FrozenSet[str],
    leet_map: Dict[str, str] = DEFAULT_LEET_MAP,
) -> Optional[str]:
    """Find the wordlist entry that best matches the alphabetic core of *password*.

    Returns the matched word (lowercase) or ``None`` if no match is found.
    """
    _, core, _ = _strip_outer_nonalpha(password)
    if not core:
        return None

    candidate = _reverse_leet(core).lower()

    if candidate in wordlist:
        return candidate

    # Also try without embedded non-alpha chars (for cores like "P@ss" → "pass").
    alpha_only = re.sub(r"[^a-zA-Z]", "", core)
    candidate2 = _reverse_leet(alpha_only).lower()
    if candidate2 and candidate2 in wordlist:
        return candidate2

    return None


# ---------------------------------------------------------------------------
# Case + leet inference
# ---------------------------------------------------------------------------

_CASE_OPS: List[Tuple[str, object]] = [
    ("",  lambda s: s),                                     # no case change
    ("c", lambda s: s[:1].upper() + s[1:].lower() if s else s),  # capitalize
    ("u", str.upper),                                        # uppercase all
    ("l", str.lower),                                        # lowercase all (no-op)
    ("t", str.swapcase),                                     # toggle all
]


def _detect_case_and_leet(
    base: str,
    core: str,
    leet_map: Dict[str, str],
) -> Optional[Tuple[str, List[str]]]:
    """Return ``(case_op, leet_sub_ops)`` that transforms *base* → *core*.

    Tries each case operation in order; for the matching op, detects leet
    substitutions char-by-char. Returns ``None`` if no combination works.

    Parameters
    ----------
    base:
        The wordlist base word (lowercase).
    core:
        The "inner" part of the password (with prefix/suffix already stripped).
    leet_map:
        ``{source_lowercase_char: leet_char}`` mapping.
    """
    if len(base) != len(core):
        return None

    for case_op, case_fn in _CASE_OPS:
        cased = case_fn(base)  # type: ignore[operator]
        if len(cased) != len(core):
            continue

        leet_ops: List[str] = []
        seen_subs: Dict[Tuple[str, str], str] = {}
        ok = True

        for cb, cc in zip(cased, core):
            if cb == cc:
                continue  # exact match — no substitution needed

            # Check if cc is the leet-substitution of cb.
            cb_lower = cb.lower()
            leet_target = leet_map.get(cb_lower)

            if leet_target is not None and leet_target == cc:
                # leet sub: sXY where X = cb (as it appears after the case op)
                sub_key = (cb, cc)
                if sub_key not in seen_subs:
                    op = f"s{cb}{cc}"
                    seen_subs[sub_key] = op
                    leet_ops.append(op)
            else:
                ok = False
                break

        if ok:
            return case_op, leet_ops

    return None


# ---------------------------------------------------------------------------
# Full rule inference
# ---------------------------------------------------------------------------

def infer_rule(
    base: str,
    password: str,
    leet_map: Dict[str, str] = DEFAULT_LEET_MAP,
) -> Optional[str]:
    """Infer a Hashcat rule string that transforms *base* into *password*.

    Returns the rule string (e.g. ``"c sa@ $1 $2 $3"``) or ``None`` if the
    transformation cannot be represented in the supported primitive subset.

    The canonical op order in the rule string is:
        [case_op] [leet_ops...] [r] [^prefix_reversed...] [$suffix...] [d]

    This order ensures that ``apply_rule(base, rule) == password``.
    """
    # -----------------------------------------------------------------------
    # Step 0: Check duplication (d).
    # -----------------------------------------------------------------------
    target = password
    dup = False
    n = len(target)
    if n >= 2 and n % 2 == 0 and target[: n // 2] == target[n // 2:]:
        target = target[: n // 2]
        dup = True

    # -----------------------------------------------------------------------
    # Step 1: Strip prefix / suffix (non-alpha outer chars).
    # -----------------------------------------------------------------------
    prefix, core, suffix = _strip_outer_nonalpha(target)

    # -----------------------------------------------------------------------
    # Step 2: Detect reversal.
    # -----------------------------------------------------------------------
    reversed_flag = False
    base_to_match = base

    # Case-insensitive comparison after stripping leet from the core.
    core_deleet = _reverse_leet(core).lower()
    if core_deleet == base[::-1]:
        reversed_flag = True
        # Work with the reversed base for case+leet detection.
        base_to_match = base[::-1]
    elif core_deleet == base:
        pass  # normal (no reversal)
    else:
        # Neither direction gives an exact alpha match — try anyway with
        # the non-reversed base (case+leet may still work on the original).
        pass

    # -----------------------------------------------------------------------
    # Step 3: Detect case op + leet substitutions.
    # -----------------------------------------------------------------------
    result = _detect_case_and_leet(base_to_match, core, leet_map)
    if result is None:
        return None

    case_op, leet_ops = result

    # -----------------------------------------------------------------------
    # Step 4: Build the rule string.
    # -----------------------------------------------------------------------
    # Prefix ops: to prepend "abc", we need ^c ^b ^a (reversed order).
    prefix_ops = [f"^{c}" for c in reversed(prefix)]
    # Suffix ops: $a $b $c
    suffix_ops = [f"${c}" for c in suffix]

    parts: List[str] = []
    if case_op:
        parts.append(case_op)
    parts.extend(leet_ops)
    if reversed_flag:
        parts.append("r")
    parts.extend(prefix_ops)
    parts.extend(suffix_ops)
    if dup:
        parts.append("d")

    rule = " ".join(parts) if parts else ":"

    # -----------------------------------------------------------------------
    # Step 5: Validate by re-applying the rule.
    # -----------------------------------------------------------------------
    if apply_rule(base, rule) != password:
        return None

    return rule


# ---------------------------------------------------------------------------
# Hashcat parity check
# ---------------------------------------------------------------------------

def parity_check(
    triples: List[Tuple[str, str, str]],
    hashcat_bin: str = "hashcat",
    timeout: int = 30,
) -> Tuple[int, int]:
    """Cross-check Python rule applier against real ``hashcat --stdout``.

    Applies each (base, rule, expected_password) triple through hashcat and
    asserts the output matches. Returns ``(passed, total)``.

    Skips silently if hashcat is not installed.
    """
    import shutil
    if not shutil.which(hashcat_bin):
        return 0, 0  # hashcat absent — test will be marked skipped upstream

    passed = 0
    for base, rule, expected in triples:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".rule", delete=False, encoding="utf-8"
        ) as rf:
            rf.write(rule + "\n")
            rf_path = rf.name
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as wf:
            wf.write(base + "\n")
            wf_path = wf.name
        try:
            res = subprocess.run(
                [hashcat_bin, "--stdout", "-r", rf_path, wf_path, "--quiet"],
                capture_output=True, text=True, timeout=timeout,
            )
            output = res.stdout.strip()
            if output == expected:
                passed += 1
        except Exception:
            pass
        finally:
            os.unlink(rf_path)
            os.unlink(wf_path)

    return passed, len(triples)
