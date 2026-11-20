"""Gera o índice de localidades a partir do ficheiro PT.zip do GeoNames."""

import argparse
import csv
import io
from pathlib import Path
from zipfile import ZipFile


def portuguese_name(name):
    return "Lisboa" if name == "Lisbon" else name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arquivo", type=Path, help="https://download.geonames.org/export/dump/PT.zip")
    parser.add_argument("--saida", type=Path, default=Path("data/meta/localidades_pt.csv"))
    args = parser.parse_args()

    with ZipFile(args.arquivo) as archive, io.TextIOWrapper(archive.open("PT.txt"), encoding="utf-8") as source:
        entries = list(csv.reader(source, delimiter="\t"))
    divisions = {
        (row[10], row[11], row[12] if row[7] == "ADM3" else ""): portuguese_name(row[1])
        for row in entries if row[7] in ("ADM2", "ADM3")
    }
    rows = []
    for row in entries:
        if row[8] != "PT" or row[6] != "P" or row[7] in ("PPLH", "PPLQ", "PPLW"):
            continue
        district, county, parish = row[10:13]
        council = divisions.get((district, county, ""), "")
        ward = divisions.get((district, county, parish), "")
        rows.append((portuguese_name(row[1]), council, ward, row[4], row[5], int(row[14] or 0)))
    rows.sort(key=lambda row: (-row[-1], row[0], row[1], row[2]))
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("name", "council", "parish", "lat", "lon", "population"))
        writer.writerows(rows)
    print(f"{len(rows)} localidades em {args.saida}")


if __name__ == "__main__":
    main()
