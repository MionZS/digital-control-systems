"""Convert impulse response TXT files in input/impulse_response to CSV and write LVM control/output pairs for identification.

Produces files under output/impulse_response/<basename>/:
 - <basename>.csv  (t,T_in,u,T_out)
 - control.lvm     (two-column LVM: time \t u)
 - output.lvm      (two-column LVM: time \t T_in)

Usage:
    uv run python Scripts/convert_impulse_txt_to_csv.py
"""

from __future__ import annotations
from pathlib import Path
import csv

IN_DIR = Path("input/impulse_response")
OUT_ROOT = Path("output/impulse_response")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TXT_FILES = [p for p in IN_DIR.glob("*.txt") if p.is_file()]

HEADER = ["t", "T_in", "u", "T_out"]

LVM_HEADER = (
    "LabVIEW Measurement\n"
    "Writer_Version\t2\n"
    "Reader_Version\t2\n"
    "Separator\tTab\n"
    "Decimal_Separator\t.\n"
    "Multi_Headings\tNo\n"
    "X_Columns\tOne\n"
    "Time_Pref\tRelative\n"
    "***End_of_Header***\n\n"
    "\tChannels\t1\n"
    "\tSamples\t{samples}\n\n"
    "X_Value\tUntitled\tComment\n"
)


def read_txt(path: Path):
    # TXT format appears to have four whitespace-separated columns per line: t, T_in, u, T_out
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        try:
            t = float(parts[0].replace(",", "."))
            Tin = float(parts[1].replace(",", "."))
            u = float(parts[2].replace(",", "."))
            Tout = float(parts[3].replace(",", "."))
            rows.append((t, Tin, u, Tout))
        except ValueError:
            continue
    return rows


def write_csv(rows, out_csv: Path):
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for r in rows:
            writer.writerow([f"{r[0]:.6f}", f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}"])


def write_lvm_two_column(rows, out_lvm: Path, col_index: int):
    # col_index: 2 for u, 1 for Tin
    samples = len(rows)
    content = LVM_HEADER.format(samples=samples)
    lines = []
    for r in rows:
        t = f"{r[0]:.6f}"
        val = f"{r[col_index]:.6f}"
        lines.append(f"{t}\t{val}\n")
    content = content + "".join(lines)
    out_lvm.write_text(content, encoding="utf-8")


def main():
    processed = []
    for txt in TXT_FILES:
        rows = read_txt(txt)
        if not rows:
            continue
        name = txt.stem
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{name}.csv"
        write_csv(rows, out_csv)
        # control.lvm: time and u (col index 2)
        write_lvm_two_column(rows, out_dir / "control.lvm", col_index=2)
        # output.lvm: time and T_in (col index 1)
        write_lvm_two_column(rows, out_dir / "output.lvm", col_index=1)
        processed.append(str(out_dir))
        print("Wrote:", out_csv)
    print("Processed folders:", processed)


if __name__ == "__main__":
    main()
