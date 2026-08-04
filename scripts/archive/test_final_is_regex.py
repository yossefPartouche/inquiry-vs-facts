import re

def extract_final_is_answer(raw_output):
    matches = re.findall(
        r"\bis\s+[\*'\"]{0,2}([\-\d./\\{}^a-zA-Z\s]+?)[\*'\"]{0,2}\.\s*(?:$|\n)",
        raw_output
    )
    if not matches:
        return None
    return matches[-1].strip()

test_cases = [
    ("...same as the 2nd letter of \"MATHLETE\", which is 'A'.", "A"),
    ("...The largest of these integers is 10.", "10"),
    ("...This is **5**.", "5"),
    ("...There are 5 such integers.", "5"),  # this one SHOULD fail — no "is"
]

for text, expected in test_cases:
    result = extract_final_is_answer(text)
    print(f"expected={expected!r}  got={result!r}  {'OK' if result==expected else 'MISMATCH'}")