# BTRC-Predictor

**Verified reproduction:** `train_btrc_model.py` in this repository has been confirmed, via direct comparison against the original analysis code and data, to reproduce the manuscript's reported result **exactly**: 68 positive / 131 negative training examples, ROC-AUC 0.755 ± 0.117, pooled out-of-fold ROC-AUC 0.729, optimal threshold 0.28, F1 0.620.

## BTRC-Predictor: A receptor-specific degron signature model for predicting BTRC (β-TrCP)-targeted substrates in the ubiquitin-proteasome system.

The recognition of substrates by E3 ubiquitin ligases determines the specificity of the ubiquitin-proteasome system; however, the prediction of E3-targeted substrates by in silico approaches is mostly confined to the general, intrinsic properties of these targets. Here we address this gap for BTRC, an F-box E3 ligase whose substrates are recognized through a crystallographically defined DSGxxS phosphodegron motif. We demonstrate that the inclusion of this receptor-specific structural constraint, in addition to substrate-intrinsic features, enables resolution of the specificity dilemma that generic, motif-agnostic approaches cannot achieve. Using a dataset of 68 verified BTRC substrates as the positive set and 131 substrates of 22 other F-box receptors as the negative set, we designed a four-feature class-weighted logistic regression + XGBoost ensemble that achieved a ROC-AUC of 0.755 +/- 0.117 under 10-fold stratified cross-validation - significantly better than motif-agnostic methods can achieve. The DSGxxS motif is enriched 12.6-fold in real BTRC substrates compared with other-ligase substrates, confirming that this receptor-specific signal is not a dataset artifact. Applying the model to a genome-wide screen of 859 proteins yields ranked candidate probabilities across known and novel BTRC substrates, for manual review and experimental follow-up.

# Installation

Download BTRC-Predictor by

```
git clone https://github.com/priyaharshiniks999/btrc-substrate-prediction.git
```

Installation has been tested with Python 3.11. Since the package is written in Python 3.x, python3.x with the pip tool must be installed. BTRC-Predictor uses the following dependencies: numpy, pandas, scikit-learn, xgboost, imbalanced-learn, biopython, requests. You can install these packages by the following commands:

```
conda create -n btrc-predictor python=3.11
conda activate btrc-predictor
pip install pandas numpy scikit-learn xgboost imbalanced-learn biopython requests
```

# Repository structure

```
data/
  shared/       Raw and processed data used by BOTH branches: the F-box protein
                family, known substrates of BTRC/SKP2/other F-box ligases (22 distinct
                ligases in the confirmed-matching negative class)
                (data/shared/known_substrates/, raw UbiBrowser search exports),
                the BTRC training feature table (data/shared/features.csv:
                DSGxxS motif, disorder score, AlphaFold pLDDT for 68 BTRC +
                131 other-ligase substrates), and the SKP2 pilot's separate
                feature table (data/shared/skp2_features.csv: confirmed
                phosphodegron features for 22 SKP2 + 284 other-ligase
                substrates - a different feature set and source from the
                BTRC table above).
  btrc/         BTRC-only genome-wide screening inputs: data/btrc/pathway_genes.csv
                (874 genes from six MSigDB Hallmark pathways) and
                data/btrc/pathway_genes_features.csv (the same four features
                computed for all 859 mapped candidates).
models/
  btrc/         The final published ensemble (lr_model.pkl, xgb_model.pkl,
                scaler.pkl, feature_columns.txt, optimal_threshold.txt = 0.28).
  skp2_pilot/   All 8 SKP2 diagnostic strategies, each with its own fitted
                model file (e.g. 1_logistic_regression.pkl) and, where the
                strategy scales its features, a matching *_scaler.pkl -
                plus strategy_comparison.csv summarizing their 10-fold CV
                ROC-AUC.
prediction/     Genome-wide screening output for BTRC: predictions and confidence
                scores (ensemble, LR, and XGBoost probabilities) for all 859
                candidate proteins, for manual review and selection rather than
                an automatic shortlist. Also holds
                prediction/skp2_pilot_predictions.csv - every SKP2-pilot
                protein's out-of-fold predicted probability from each of the
                8 modeling strategies (i.e. each protein's prediction from
                the one CV fold where it was held out and never seen during
                that fold's training).
```

Five scripts, run in order:

| Script | What it does |
|---|---|
| `fetch_data.py` | Collects the F-box protein family from UniProt and parses known-substrate evidence into `other_ligase_substrates.csv` |
| `build_features.py` | Computes the DSGxxS motif scan, IUPred2A disorder score, and AlphaFold pLDDT for every protein |
| `train_btrc_model.py` | **Trains the final BTRC ensemble** - the paper's actual published result (ROC-AUC 0.755, threshold 0.28) |
| `train_skp2_pilot.py` | Reproduces the SKP2 diagnostic: 8 modeling strategies on confirmed phosphodegron features (22 positives / 284 negatives, 5-fold CV), showing the ~0.58-0.60 ceiling that motivates the BTRC approach |
| `screen_genome_wide.py` | Applies the trained BTRC model to 859 proteins from six MSigDB Hallmark pathways, producing per-protein confidence scores for manual review |

# Usage

```
python fetch_data.py
python build_features.py
python train_btrc_model.py
python train_skp2_pilot.py      # optional: reproduces the motivating SKP2 diagnostic
python screen_genome_wide.py
python verify.py                # cross-checks everything above against the manuscript's exact numbers
```

`data/`, `models/`, and `prediction/` already contain the real, precomputed results from the published analysis, so each script can also be inspected or re-run independently without repeating the full pipeline from scratch.

# Results

Applying the trained BTRC ensemble to a genome-wide screen of 859 proteins drawn from six MSigDB Hallmark pathways produces a full ranked table of ensemble, logistic regression, and XGBoost probabilities for every candidate (`prediction/genome_wide_predictions.csv`). Candidates for experimental follow-up are selected manually from this table based on confidence and structural tractability, rather than by an automatic cutoff.

# Citation

Please cite the following paper if you use this code or data: *Beyond Substrate-Intrinsic Features: A Receptor-Specific Degron Signature Enables Accurate Prediction of BTRC (beta-TrCP)-Targeted Substrates in the Ubiquitin-Proteasome System.* (citation details to be added upon publication)

# Data sources

- UniProt: https://www.uniprot.org
- AlphaFold Protein Structure Database: https://alphafold.ebi.ac.uk
- MSigDB Hallmark gene sets: https://www.gsea-msigdb.org/gsea/msigdb
- IUPred2A: https://iupred2a.elte.hu
- UbiBrowser: http://ubibrowser.bio-it.cn

# License

See `LICENSE`.
