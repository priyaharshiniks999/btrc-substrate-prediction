"""
screen_genome_wide.py
======================
Applies the trained BTRC classifier to a genome-wide candidate set:
proteins from six MSigDB Hallmark pathways implicated in BTRC-relevant
biology (cell cycle checkpoints, growth signaling, apoptosis, DNA
repair, Wnt signaling), to nominate novel candidate substrates.

Pathways (see manuscript, Results 3.5):
  G2M_CHECKPOINT, MTORC1_SIGNALING, TNFA_SIGNALING_VIA_NFKB,
  APOPTOSIS, DNA_REPAIR, WNT_BETA_CATENIN_SIGNALING

Steps:
  1. Fetch the six Hallmark gene sets from MSigDB and merge/deduplicate.
  2. Map each gene symbol to a UniProt accession.
  3. Compute the same four features used in training (build_features.py).
  4. Score every candidate with the trained LR + XGBoost ensemble.

Output: prediction/genome_wide_predictions.csv
        (gene_name, accession, lr_probability, xgb_probability,
         ensemble_probability)

Run from the repository root, after train_btrc_model.py:
    python screen_genome_wide.py
"""

import os
import csv
import pickle
import time
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "btrc")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "btrc")
PRED_DIR = os.path.join(os.path.dirname(__file__), "prediction")

GMT_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/h.all.v2023.2.Hs.symbols.gmt"
PATHWAYS = [
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MTORC1_SIGNALING",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_DNA_REPAIR",
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
]
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

# import feature-computation helpers from build_features.py
from build_features import scan_dsgxxs_motif, fetch_disorder_score, fetch_mean_plddt


def fetch_pathway_genes():
    """Download the Hallmark .gmt file and return the deduplicated union
    of genes across the six pathways above."""
    r = requests.get(GMT_URL, timeout=60)
    r.raise_for_status()
    gene_sets = {}
    for line in r.text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        name, _desc, *genes = parts
        gene_sets[name] = genes

    merged = set()
    for pathway in PATHWAYS:
        merged.update(gene_sets.get(pathway, []))
    return sorted(merged)


def map_genes_to_accessions(genes):
    """Resolve each gene symbol to a reviewed human UniProt accession
    and sequence."""
    records = []
    for gene in genes:
        params = {
            "query": f"(gene:{gene}) AND (organism_id:9606) AND (reviewed:true)",
            "fields": "accession,sequence",
            "format": "json",
            "size": 1,
        }
        r = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=15)
        hits = r.json().get("results", [])
        if hits:
            records.append(
                {
                    "gene_name": gene,
                    "accession": hits[0]["primaryAccession"],
                    "sequence": hits[0].get("sequence", {}).get("value", ""),
                }
            )
        time.sleep(0.1)
    return records


def compute_candidate_features(records):
    for rec in records:
        has_motif, motif_count = scan_dsgxxs_motif(rec["sequence"])
        rec["has_dsgxxs_motif"] = has_motif
        rec["dsgxxs_count"] = motif_count
        rec["max_disorder_score"] = fetch_disorder_score(rec["sequence"])
        rec["mean_plddt"] = fetch_mean_plddt(rec["accession"])
    return records


def load_models():
    with open(os.path.join(MODEL_DIR, "lr_model.pkl"), "rb") as f:
        lr = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_columns.txt")) as f:
        feature_cols = f.read().splitlines()
    return lr, xgb_model, scaler, feature_cols


def score_candidates(records, lr, xgb_model, scaler, feature_cols):
    df = pd.DataFrame(records).dropna(subset=feature_cols)
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    df["lr_probability"] = lr.predict_proba(X_scaled)[:, 1]
    df["xgb_probability"] = xgb_model.predict_proba(X)[:, 1]
    df["ensemble_probability"] = (df["lr_probability"] + df["xgb_probability"]) / 2
    return df


if __name__ == "__main__":
    genes = fetch_pathway_genes()
    print(f"[screen_genome_wide] {len(genes)} genes across 6 MSigDB Hallmark pathways")
    with open(os.path.join(DATA_DIR, "pathway_genes.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gene_name"])
        for g in genes:
            writer.writerow([g])

    records = map_genes_to_accessions(genes)
    print(f"[screen_genome_wide] {len(records)} genes mapped to UniProt accessions")
    pd.DataFrame(records)[["gene_name", "accession"]].to_csv(
        os.path.join(DATA_DIR, "pathway_genes_accessions.csv"), index=False
    )

    records = compute_candidate_features(records)
    feature_cols_preview = ["gene_name", "accession", "has_dsgxxs_motif", "dsgxxs_count",
                             "max_disorder_score", "mean_plddt"]
    pd.DataFrame(records)[feature_cols_preview].to_csv(
        os.path.join(DATA_DIR, "pathway_genes_features.csv"), index=False
    )
    print(f"[screen_genome_wide] Candidate features saved to data/pathway_genes_features.csv")

    lr, xgb_model, scaler, feature_cols = load_models()
    scored = score_candidates(records, lr, xgb_model, scaler, feature_cols)

    os.makedirs(PRED_DIR, exist_ok=True)
    out_path = os.path.join(PRED_DIR, "genome_wide_predictions.csv")
    scored[["gene_name", "accession", "lr_probability", "xgb_probability", "ensemble_probability"]].to_csv(
        out_path, index=False
    )
    print(f"[screen_genome_wide] {len(scored)} candidates scored -> {out_path}")
