"""
fetch_data.py
=============
Collects the raw protein and substrate data needed for BTRC substrate
prediction:

  1. The human F-box protein family (from UniProt, by Pfam domain PF00646).
  2. Protein sequences for the F-box family (FASTA, from UniProt).
  3. Known substrates of BTRC and 20 other F-box ligases, parsed from
     UbiBrowser search exports (see parse_ubibrowser_exports() below) -
     these exports must be downloaded manually from http://ubibrowser.bio-it.cn
     by searching each ligase name and saving the results table as a
     tab-separated .txt file into data/shared/known_substrates/.

Output (written to data/):
  - fbox_family_proteins.csv   (accession, gene_name, protein_name)
  - fbox_family_sequences.fasta
  - other_ligase_substrates.csv (gene_name, source_ligase, uniprot_accession)

Run from the repository root:
    python fetch_data.py
"""

import os
import csv
import glob
import time
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "shared")
KNOWN_SUBSTRATES_DIR = os.path.join(DATA_DIR, "known_substrates")

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
FBOX_PFAM_DOMAIN = "PF00646"  # F-box domain


def fetch_fbox_family(organism_id="9606"):
    """Query UniProt for all reviewed human proteins carrying an F-box
    domain (Pfam PF00646). Returns a list of dicts with accession,
    gene_name, and protein_name."""
    params = {
        "query": f"(xref:pfam-{FBOX_PFAM_DOMAIN}) AND (organism_id:{organism_id}) AND (reviewed:true)",
        "fields": "accession,gene_names,protein_name,sequence",
        "format": "json",
        "size": 500,
    }
    r = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])

    proteins = []
    for entry in results:
        accession = entry["primaryAccession"]
        genes = entry.get("genes", [])
        gene_name = genes[0]["geneName"]["value"] if genes else accession
        protein_name = (
            entry.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", "")
        )
        sequence = entry.get("sequence", {}).get("value", "")
        proteins.append(
            {
                "accession": accession,
                "gene_name": gene_name,
                "protein_name": protein_name,
                "sequence": sequence,
            }
        )
    return proteins


def save_fbox_family(proteins):
    os.makedirs(DATA_DIR, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, "fbox_family_proteins.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["accession", "gene_name", "protein_name"])
        writer.writeheader()
        for p in proteins:
            writer.writerow(
                {k: p[k] for k in ["accession", "gene_name", "protein_name"]}
            )

    fasta_path = os.path.join(DATA_DIR, "fbox_family_sequences.fasta")
    with open(fasta_path, "w", encoding="utf-8") as f:
        for p in proteins:
            if p["sequence"]:
                f.write(f">{p['accession']}|{p['gene_name']}\n{p['sequence']}\n")

    print(f"[fetch_fbox_family] {len(proteins)} proteins -> {csv_path}")


def parse_ubibrowser_exports():
    """Parse UbiBrowser search-result exports (one .txt file per ligase,
    tab-separated: Species / SUBGENE / E3GENE / SOURCE / SOURCEID / SENTENCE)
    from data/known_substrates/*.txt into a single substrate table.

    UniProt accessions are not included in UbiBrowser's export and must be
    resolved separately (see resolve_accessions() below) before this table
    is used as the negative training class.
    """
    rows = []
    export_files = sorted(glob.glob(os.path.join(KNOWN_SUBSTRATES_DIR, "*.txt")))
    if not export_files:
        print(
            f"[parse_ubibrowser_exports] No files found in {KNOWN_SUBSTRATES_DIR}. "
            "Download UbiBrowser search exports for each ligase first."
        )
        return rows

    for path in export_files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                gene_name, e3_gene = parts[1], parts[2]
                rows.append({"gene_name": gene_name, "source_ligase": e3_gene})
    return rows


def resolve_accessions(rows):
    """Look up UniProt accessions for each (gene_name) pair via the
    UniProt search API. Cached per gene to minimize API calls."""
    cache = {}
    resolved = []
    for row in rows:
        gene = row["gene_name"]
        if gene not in cache:
            params = {
                "query": f"(gene:{gene}) AND (organism_id:9606) AND (reviewed:true)",
                "fields": "accession",
                "format": "json",
                "size": 1,
            }
            r = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=15)
            hits = r.json().get("results", [])
            cache[gene] = hits[0]["primaryAccession"] if hits else None
            time.sleep(0.1)  # be polite to the API
        row["uniprot_accession"] = cache[gene]
        if row["uniprot_accession"]:
            resolved.append(row)
    return resolved


def save_other_ligase_substrates(rows):
    """Save ALL collected substrate rows - including BTRC's own substrates -
    to one combined table. The positive (BTRC) vs. negative (other ligases)
    split happens later, at labeling time in train_btrc_model.py and
    train_skp2_pilot.py, not here. (An earlier version of this script
    excluded BTRC at this stage, which was wrong: the real data collection
    pulled BTRC's substrates from the same known_substrates/ UbiBrowser
    exports as the other 20 ligases - there was no separate curation step.)"""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "other_ligase_substrates.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gene_name", "source_ligase", "uniprot_accession"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save_other_ligase_substrates] {len(rows)} rows -> {csv_path}")


if __name__ == "__main__":
    proteins = fetch_fbox_family()
    save_fbox_family(proteins)

    raw_rows = parse_ubibrowser_exports()
    if raw_rows:
        resolved_rows = resolve_accessions(raw_rows)
        save_other_ligase_substrates(resolved_rows)
