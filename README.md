# SeqEngine: Automated Feature Engineering & Selection for Protein and DNA Sequences

[![PyPI version](https://badge.fury.io/py/seqengine.svg)](https://badge.fury.io/py/seqengine)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SeqEngine** is a fully automated, end‑to‑end pipeline for feature extraction and feature selection from biological sequences (protein and DNA). It extracts a comprehensive set of **610 features for proteins** and **400 features for DNA**, implements **seven feature selection methods** (univariate filters, wrapper, and embedded), and outputs multiple curated feature subsets along with a detailed text report containing sequence statistics, feature rankings, and performance metrics. Designed for researchers who need a reproducible, interpretable, and efficient tool to pre‑process sequence data for machine learning.

---

## 📊 Pipeline Overview

The figure below shows the complete workflow of SeqEngine:

![SeqEngine Workflow](Workflow.png)

---

## ✨ Key Features

- **Comprehensive Feature Extraction**:
  - **Protein**: 610 features including AAC, dipeptide, PAAC, CTD, Moran autocorrelation, BLOSUM62, and physicochemical properties.
  - **DNA**: 400 features including NAC, DNC, TNC, k‑mer (k=4), PseDNC, Moran autocorrelation, and ENAC.
- **Multiple Feature Selection Strategies**:
  - **Univariate Filters**: `f_classif`, `mutual_info`, `chi2`
  - **Wrapper**: Recursive Feature Elimination (RFE) with Random Forest
  - **Embedded**: Random Forest importance, Lasso (L1), Gradient Boosting importance
- **Human‑Readable Output**: Feature names like `AAC_Alanine`, `CTD_Comp_Hydrophobicity_G1`, `Kmer_AAAA`—no more cryptic `F1`, `F2`.
- **Batch Processing** with `tqdm` progress bars for large datasets.
- **Parallel Execution** using multiple CPU cores.
- **Comprehensive Report**: Sequence statistics, feature statistics, method‑by‑method performance (accuracy, precision, recall, F1, MCC), top‑20 ranked features, and execution times.
- **Multiple Output Formats**: Consolidated feature CSV, per‑method selected feature CSVs, aggregate ranking summary, and a full text report.

---

## 📦 Installation

```bash
pip install seqengine
