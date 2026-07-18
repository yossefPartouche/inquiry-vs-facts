"""
Assigns a problem_type tag from the fixed closed set used by
prompts/fact_library.json. Keyword-matching first pass — not a classifier.
"""

VALID_TYPES = (
    "modular_exponentiation",
    "divisibility",
    "gcd_lcm",
    "primes_factorization",
    "base_conversion",
)

# Checked in this order. First match wins, so order encodes priority
# for problems whose text triggers more than one category's keywords.
#
# modular_exponentiation is a special case: it requires BOTH a mod signal
# AND an exponent signal. A bare "remainder"/"mod" problem (e.g. "sum
# divided by 8") is plain modular arithmetic, not modular exponentiation,
# and must NOT be tagged this way -- it would cause B1/B2 to inject
# irrelevant Fermat's-Little-Theorem-style facts for a problem that has
# no exponent in it at all. Found via manual spot-check on the real
# Number Theory L1-2 pool (300 problems); fixed after inspecting samples.
_MOD_SIGNALS = ["mod ", "modulo", "remainder", "\\pmod", "congruent"]
_EXPONENT_SIGNALS = ["^", "\\^", "power of", "exponent", "raised to"]

_RULES = [
    ("base_conversion", ["base ", "base-", "base_", "binary", "octal", "hexadecimal"]),
    ("gcd_lcm", ["gcd", "greatest common divisor", "lcm", "least common multiple"]),
    # modular_exponentiation handled separately below, not via plain _RULES
    ("primes_factorization", ["prime", "factor", "factoriz", "factoris"]),
    ("divisibility", ["divisible", "divisor", "divides"]),
]


def _has_any(text, keywords):
    return any(kw in text for kw in keywords)


def tag_problem_type(problem_text):
    """
    Returns one of VALID_TYPES. Never returns None/blank -- if nothing
    matches, falls back to the closest category (divisibility, the
    broadest bucket) and the caller should flag these for manual review.
    """
    text = problem_text.lower()

    for problem_type, keywords in _RULES:
        if problem_type == "primes_factorization":
            # modular_exponentiation is checked here, before
            # primes_factorization, to preserve the original priority order
            # (base_conversion -> gcd_lcm -> modular_exponentiation ->
            # primes_factorization -> divisibility).
            if _has_any(text, _MOD_SIGNALS) and _has_any(text, _EXPONENT_SIGNALS):
                return "modular_exponentiation"
        if _has_any(text, keywords):
            return problem_type

    # FLAG FOR MANUAL REVIEW: no keyword matched, defaulted to closest
    # available category rather than inventing a 6th one.
    return "divisibility"