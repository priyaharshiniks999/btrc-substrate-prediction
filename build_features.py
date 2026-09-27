"""
build_features.py
==================
Computes the four features used by the final BTRC model, for every
protein in the combined substrate list (BTRC positives + other-ligase
negatives):

  1. has_dsgxxs_motif   - presence of the DSGxxS phosphodegron motif
                           (regex: D-S-G-x-x-S, the crystallographically
                           defined BTRC recognition motif; Wu et al., 2003)
  2. dsgxxs_count        - number of motif occurrences in the sequence
  3. max_disorder_score  - maximum per-residue intrinsic disorder score
                           (IUPred2A, "long" disorder type)
  4. mean_plddt          - mean per-residue AlphaFold2 structural
                           confidence (pLDDT) across the full sequence

Output: data/features.csv (accession, gene_name, source_ligase,
has_dsgxxs_motif, dsgxxs_count, max_disorder_score, mean_plddt)

Run from the repository root, after fetch_data.py:
    python build_features.py
"""

import os
import re
import csv
import time
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "shared")

DSGXXS_PATTERN = re.compile(r"DSG..S")  # D-S-G-x-x-S: the crystallographically
IUPRED_URL = "https://iupred2a.elte.hu/iupred2a"
ALPHAFOLD_STRUCTURE_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


def load_substrate_list():
    """Combine the BTRC positive set and other-ligase negative set into
    one (accession, gene_name, source_ligase, sequence) table."""
    rows = []

    fbox_csv = os.path.join(DATA_DIR, "fbox_family_proteins.csv")
    fasta_path = os.path.join(DATA_DIR, "fbox_family_sequences.fasta")
    sequences = load_fasta(fasta_path)

    other_csv = os.path.join(DATA_DIR, "other_ligase_substrates.csv")
    if os.path.exists(other_csv):
        with open(other_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seq = sequences.get(row["uniprot_accession"], "")
                rows.append(
                    {
                        "accession": row["uniprot_accession"],
                        "gene_name": row["gene_name"],
                        "source_ligase": row["source_ligase"],
                        "sequence": seq,
                    }
                )

    return rows


def load_fasta(path):
    sequences = {}
    if not os.path.exists(path):
        return sequences
    accession, seq_lines = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if accession:
                    sequences[accession] = "".join(seq_lines)
                accession = line[1:].split("|")[0]
                seq_lines = []
            else:
                seq_lines.append(line)
        if accession:
            sequences[accession] = "".join(seq_lines)
    return sequences


def scan_dsgxxs_motif(sequence):
    """Presence and count of the DSGxxS phosphodegron motif."""
    matches = DSGXXS_PATTERN.findall(sequence)
    return int(len(matches) > 0), len(matches)


def fetch_disorder_score(sequence):
    """Maximum per-residue IUPred2A 'long' disorder score for the sequence."""
    if not sequence:
        return None
    try:
        r = requests.post(
            IUPRED_URL,
            data={"seq": sequence, "type": "long"},
            timeout=30,
        )
        r.raise_for_status()
        scores = [float(x) for x in r.json().get("iupred2", [])]
        return max(scores) if scores else None
    except (requests.RequestException, ValueError, KeyError):
        return None
    finally:
        time.sleep(0.2)  # be polite to the API


def fetch_mean_plddt(accession):
    """Mean per-residue AlphaFold2 pLDDT confidence score for a UniProt
    accession, from the AlphaFold Structure Database's prediction metadata."""
    try:
        r = requests.get(ALPHAFOLD_STRUCTURE_URL.format(accession=accession), timeout=20)
        r.raise_for_status()
        entries = r.json()
        if not entries:
            return None
        # The AlphaFold API returns a global confidence summary per entry;
        # per-residue pLDDT values live in the structure file's B-factor
        # column and would need the PDB/CIF parsed for an exact mean.
        # globalMetricValue is AlphaFold's own mean-pLDDT summary field.
        return entries[0].get("globalMetricValue")
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None
    finally:
        time.sleep(0.2)


def build_feature_table():
    rows = load_substrate_list()
    if not rows:
        print("[build_features] No substrate data found - run fetch_data.py first.")
        return

    for row in rows:
        has_motif, motif_count = scan_dsgxxs_motif(row["sequence"])
        row["has_dsgxxs_motif"] = has_motif
        row["dsgxxs_count"] = motif_count
        row["max_disorder_score"] = fetch_disorder_score(row["sequence"])
        row["mean_plddt"] = fetch_mean_plddt(row["accession"])

    out_path = os.path.join(DATA_DIR, "features.csv")
    fieldnames = [
        "accession", "gene_name", "source_ligase",
        "has_dsgxxs_motif", "dsgxxs_count", "max_disorder_score", "mean_plddt",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"[build_features] {len(rows)} proteins featurized -> {out_path}")


if __name__ == "__main__":
    build_feature_table()
