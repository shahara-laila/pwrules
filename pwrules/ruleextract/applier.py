"""Python implementation of the Hashcat rule engine (subset).

Supports the primitives used by this project's rule extraction:
    :  l  u  c  C  t  T[N]  r  d  f  {  }  [  ]  k  K  q
    $[x]  ^[x]  D[N]  z[N]  Z[N]  y[N]  Y[N]  @[x]
    p[N]  s[XY]  i[NX]  o[NX]  *[NM]  x[NM]  +[N]  -[N]  L[N]  R[N]

Position parameters N, M are single characters: '0'-'9' or 'A'-'Z'/'a'-'z'
representing positions 0-35 (matching Hashcat's CONV_POS convention).

Each function in the public API:

    tokenize_rule(rule_str)  →  list of (fn_char, param_str) tuples
    apply_function(word, fn, params)  →  transformed word
    apply_rule(word, rule_str)  →  word after applying all functions in rule
"""

from __future__ import annotations

from typing import List, Tuple


# ---------------------------------------------------------------------------
# Position encoding helpers
# ---------------------------------------------------------------------------

def _conv_pos(c: str) -> int:
    """Convert a single position/length char to an int (Hashcat convention)."""
    if not c:
        return 0
    if c.isdigit():
        return int(c)
    if c.isupper():
        return ord(c) - ord("A") + 10
    if c.islower():
        return ord(c) - ord("a") + 10
    return 0


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

# Functions that consume 0 additional chars (function only).
_ZERO_PARAM = set(":lucCtrdpf{}[]kKq")

# Functions that consume 1 additional char (function + 1 param char).
_ONE_PARAM = set("$^TDzZyY@+-LRp")

# Functions that consume 2 additional chars (function + 2 param chars).
_TWO_PARAM = set("sio*x")


def tokenize_rule(rule_str: str) -> List[Tuple[str, str]]:
    """Parse a Hashcat rule string into a list of ``(function, params)`` tuples.

    Hashcat rule strings are whitespace-separated tokens; each token's first
    character is the function name, followed immediately (no space) by zero,
    one, or two parameter characters.

    Examples
    --------
    >>> tokenize_rule("c sa@ $1 $2 $3")
    [('c', ''), ('s', 'a@'), ('$', '1'), ('$', '2'), ('$', '3')]
    >>> tokenize_rule("r")
    [('r', '')]
    >>> tokenize_rule(":")
    [(':', '')]
    """
    tokens: List[Tuple[str, str]] = []
    for tok in rule_str.split():
        if not tok:
            continue
        fn = tok[0]
        params = tok[1:]
        tokens.append((fn, params))
    return tokens


# ---------------------------------------------------------------------------
# Single-function applier
# ---------------------------------------------------------------------------

def apply_function(word: str, fn: str, params: str) -> str:  # noqa: C901
    """Apply a single Hashcat rule function to *word* and return the result.

    Unknown functions are silently treated as no-ops (mirrors Hashcat's
    behaviour for unrecognised rule chars).
    """
    if fn == ":":
        return word

    # --- Case functions ---
    if fn == "l":
        return word.lower()
    if fn == "u":
        return word.upper()
    if fn == "c":
        return (word[0].upper() + word[1:].lower()) if word else word
    if fn == "C":
        return (word[0].lower() + word[1:].upper()) if word else word
    if fn == "t":
        return word.swapcase()
    if fn == "T":
        if not params:
            return word
        pos = _conv_pos(params[0])
        chars = list(word)
        if 0 <= pos < len(chars):
            chars[pos] = chars[pos].swapcase()
        return "".join(chars)

    # --- Structural functions ---
    if fn == "r":
        return word[::-1]
    if fn == "d":
        return word + word
    if fn == "f":
        return word + word[::-1]
    if fn == "{":
        return (word[1:] + word[0]) if word else word
    if fn == "}":
        return (word[-1] + word[:-1]) if word else word
    if fn == "[":
        return word[1:]
    if fn == "]":
        return word[:-1]
    if fn == "k":
        return (word[1] + word[0] + word[2:]) if len(word) >= 2 else word
    if fn == "K":
        return (word[:-2] + word[-1] + word[-2]) if len(word) >= 2 else word
    if fn == "q":
        return "".join(c * 2 for c in word)
    if fn == "p":
        if not params:
            return word
        n = _conv_pos(params[0])
        return word * n if n > 0 else word

    # --- Append / prepend ---
    if fn == "$":
        return word + params[0] if params else word
    if fn == "^":
        return params[0] + word if params else word

    # --- Position-based single-char ops ---
    if fn == "D":
        if not params:
            return word
        pos = _conv_pos(params[0])
        chars = list(word)
        if 0 <= pos < len(chars):
            del chars[pos]
        return "".join(chars)

    if fn == "z":
        if not params:
            return word
        n = _conv_pos(params[0])
        return word[0] * n + word if word else word

    if fn == "Z":
        if not params:
            return word
        n = _conv_pos(params[0])
        return word + word[-1] * n if word else word

    if fn == "y":
        if not params:
            return word
        n = _conv_pos(params[0])
        return word[:n] + word

    if fn == "Y":
        if not params:
            return word
        n = _conv_pos(params[0])
        tail = word[-n:] if n <= len(word) else word
        return word + tail

    if fn == "@":
        return word.replace(params[0], "") if params else word

    if fn == "+":
        if not params:
            return word
        pos = _conv_pos(params[0])
        chars = list(word)
        if 0 <= pos < len(chars):
            chars[pos] = chr((ord(chars[pos]) + 1) & 0xFF)
        return "".join(chars)

    if fn == "-":
        if not params:
            return word
        pos = _conv_pos(params[0])
        chars = list(word)
        if 0 <= pos < len(chars):
            chars[pos] = chr((ord(chars[pos]) - 1) & 0xFF)
        return "".join(chars)

    if fn == "L":
        if not params:
            return word
        pos = _conv_pos(params[0])
        chars = list(word)
        if 0 <= pos < len(chars):
            chars[pos] = chr((ord(chars[pos]) << 1) & 0xFF)
        return "".join(chars)

    if fn == "R":
        if not params:
            return word
        pos = _conv_pos(params[0])
        chars = list(word)
        if 0 <= pos < len(chars):
            chars[pos] = chr((ord(chars[pos]) >> 1) & 0xFF)
        return "".join(chars)

    # --- Two-param ops ---
    if fn == "s":
        if len(params) < 2:
            return word
        return word.replace(params[0], params[1])

    if fn == "i":
        if len(params) < 2:
            return word
        pos = _conv_pos(params[0])
        char = params[1]
        return word[:pos] + char + word[pos:]

    if fn == "o":
        if len(params) < 2:
            return word
        pos = _conv_pos(params[0])
        char = params[1]
        chars = list(word)
        if 0 <= pos < len(chars):
            chars[pos] = char
        return "".join(chars)

    if fn == "*":
        if len(params) < 2:
            return word
        p1 = _conv_pos(params[0])
        p2 = _conv_pos(params[1])
        chars = list(word)
        if 0 <= p1 < len(chars) and 0 <= p2 < len(chars):
            chars[p1], chars[p2] = chars[p2], chars[p1]
        return "".join(chars)

    if fn == "x":
        if len(params) < 2:
            return word
        start = _conv_pos(params[0])
        length = _conv_pos(params[1])
        return word[start: start + length]

    # Unknown → passthrough.
    return word


# ---------------------------------------------------------------------------
# Full rule applier
# ---------------------------------------------------------------------------

def apply_rule(word: str, rule_str: str) -> str:
    """Apply a complete Hashcat rule string to *word*.

    Parameters
    ----------
    word:
        Input word (the base word from the attack wordlist).
    rule_str:
        Whitespace-separated Hashcat rule string, e.g. ``"c sa@ $1 $2 $3"``.

    Returns
    -------
    The transformed candidate string.

    Examples
    --------
    >>> apply_rule("password", "c sa@ $1 $2 $3")
    'P@ssword123'
    >>> apply_rule("dragon", "r")
    'nogard'
    >>> apply_rule("abc", "d")
    'abcabc'
    """
    result = word
    for fn, params in tokenize_rule(rule_str):
        result = apply_function(result, fn, params)
    return result
