"""Write a single-file version of the manuscript for journal submission.

Springer Nature asks for one .tex file, with figures supplied separately. This script inlines
numbers.tex and the generated tables, points figures at a local figures/ folder, and copies the
class file, bibliography style, bibliography, compiled .bbl and figure PDFs into submission/,
then compiles it there as a check and zips the folder.

Usage (after latexmk has built main.pdf):  python3 make_submission.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "submission"


def main() -> None:
    src = (HERE / "main.tex").read_text()
    numbers = (HERE / "numbers.tex").read_text()
    src = src.replace("\\input{numbers.tex}", numbers.strip())

    def inline_table(m: re.Match) -> str:
        return (ROOT / "tables" / "tex" / f"{m.group(1)}.tex").read_text().strip()

    src = re.sub(r"\\input\{\\tabpath/(tab_[a-z]+)\.tex\}", inline_table, src)
    src = src.replace("\\newcommand{\\figpath}{../figures/pdf}", "\\newcommand{\\figpath}{figures}")
    src = src.replace("\\newcommand{\\tabpath}{../tables/tex}\n", "")
    assert "\\input" not in src, "an \\input was left in the flattened file"

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)
    (OUT / "main.tex").write_text(src)
    for name in ("sn-jnl.cls", "sn-basic.bst", "references.bib", "main.bbl"):
        shutil.copy(HERE / name, OUT / name)
    for fig in sorted(set(re.findall(r"\\figpath/([\w-]+\.pdf)", src))):
        shutil.copy(ROOT / "figures" / "pdf" / fig, OUT / "figures" / fig)

    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", "main.tex"],
                   cwd=OUT, check=True, stdout=subprocess.DEVNULL)
    for ext in ("aux", "blg", "fdb_latexmk", "fls", "log", "out"):
        (OUT / f"main.{ext}").unlink(missing_ok=True)
    shutil.make_archive(str(HERE / "submission"), "zip", OUT)
    print(f"wrote {OUT} and {HERE / 'submission.zip'}")


if __name__ == "__main__":
    main()
