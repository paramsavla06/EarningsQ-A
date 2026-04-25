import re

aliases = ['consolidated income', 'total income', 'income']
label_regex = '|'.join(sorted((re.escape(a) for a in aliases), key=len, reverse=True))

fwd = rf'(?:{label_regex}).{{0,150}}?(?:was|is|at|of|stood at|reported|came in at|amounted to|totaled|totalled)?.{{0,60}}?(?:inr|rs\.?|rs|nr)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|crore|crores|bn|billion)?'
bwd = rf'(?:inr|rs\.?|rs|nr)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|crore|crores|bn|billion).{{0,100}}?(?:{label_regex})'

test_sentences = [
    'Total income for Q1 FY2025 grew by 14% to INR 8,277 million',
    'Our total income stood at Rs. 2,500 crore for Q2',
    'Total income came in at INR 1,234 million',
    'Total income: INR 10,000 million',
    'The company reported total income of 8277 million',
    'INR 8,277 million was the total income for the quarter',
    'Total income in Q2 FY25 was INR 5,000 crore',
    'We had total income of Rs 2500 crore this quarter',
]

all_pass = True
for s in test_sentences:
    m = re.search(fwd, s, re.IGNORECASE | re.DOTALL)
    b = re.search(bwd, s, re.IGNORECASE | re.DOTALL)
    result = m.group(1, 2) if m else (b.group(1, 2) if b else None)
    status = "OK  " if result else "MISS"
    print(f"  [{status}] {s}")
    if result:
        print(f"         -> amount={result[0]}, unit={result[1]}")
    else:
        all_pass = False

print()
print("ALL PASS" if all_pass else "SOME TESTS FAILED")
