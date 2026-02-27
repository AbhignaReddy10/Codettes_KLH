import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP = {"fix_csv_paths.py"}

# Matches "something.csv" or 'something.csv'
# Skips:
# - already prefixed with datasets/
# - absolute paths (C:\..., /..., \...)
# - relative ./ or ../ paths
PATTERN = re.compile(
    r"""(?P<q>['"])(?P<p>(?!datasets[\\/])(?!(?:[A-Za-z]:[\\/]|[\\/]|\.{1,2}[\\/]))[^'"]+?\.csv)(?P=q)""",
    re.IGNORECASE,
)

def repl(m: re.Match) -> str:
    q = m.group("q")
    p = m.group("p")
    # Only prefix simple filenames (no folder separators)
    if "/" in p or "\\" in p:
        return m.group(0)
    return f"{q}datasets/{p}{q}"

changed_files = []

for py_file in ROOT.glob("*.py"):
    if py_file.name in SKIP:
        continue

    text = py_file.read_text(encoding="utf-8")
    new_text, count = PATTERN.subn(repl, text)

    if count > 0 and new_text != text:
        py_file.write_text(new_text, encoding="utf-8")
        changed_files.append((py_file.name, count))

if not changed_files:
    print("No CSV path updates were needed.")
else:
    print("Updated files:")
    for name, count in changed_files:
        print(f" - {name}: {count} replacement(s)")
