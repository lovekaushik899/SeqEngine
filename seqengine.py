#!/usr/bin/env python3

import os
import sys
import argparse
import time
import warnings
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef
)
from scipy import stats
from Bio import SeqIO
from Bio.SeqUtils import ProtParam, molecular_weight

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'
AA_NAMES = {
    'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'AsparticAcid',
    'C': 'Cysteine', 'Q': 'Glutamine', 'E': 'GlutamicAcid', 'G': 'Glycine',
    'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine', 'K': 'Lysine',
    'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine',
    'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine'
}
NUCS = 'ACGT'

# Feature selection methods
FS_METHODS = {
    'f_classif': {'class': SelectKBest, 'params': {'score_func': f_classif}, 'type': 'univariate'},
    'mutual_info': {'class': SelectKBest, 'params': {'score_func': mutual_info_classif}, 'type': 'univariate'},
    'chi2': {'class': SelectKBest, 'params': {'score_func': chi2}, 'type': 'univariate'},
    'rfe_rf': {'class': RFE, 'params': {'estimator': RandomForestClassifier(n_estimators=100, random_state=42)}, 'type': 'wrapper'},
    'rf_importance': {'class': SelectFromModel, 'params': {'estimator': RandomForestClassifier(n_estimators=100, random_state=42)}, 'type': 'embedded'},
    'lasso': {'class': SelectFromModel, 'params': {'estimator': LassoCV(cv=5, random_state=42)}, 'type': 'embedded'},
    'gb_importance': {'class': SelectFromModel, 'params': {'estimator': GradientBoostingClassifier(n_estimators=100, random_state=42)}, 'type': 'embedded'},
}

# ============================================================================
# FEATURE EXTRACTION WITH DESCRIPTIVE NAMES
# ============================================================================

class FeatureExtractor:
    """Extract features with descriptive names for protein and DNA sequences."""

    def __init__(self, seq_type: str, sequences: List[Tuple[str, str, int]]):
        self.seq_type = seq_type
        self.sequences = sequences

    def extract_all(self, batch_size: int = 1000) -> pd.DataFrame:
        """Extract all features, returning a DataFrame with named columns."""
        # Generate feature names from a dummy sequence
        dummy_seq = 'A' * 30 if self.seq_type == 'prot' else 'A' * 30
        _, feature_names = self._extract_single(dummy_seq, return_names=True)

        all_features = []
        total = len(self.sequences)

        with tqdm(total=total, desc="Extracting features", unit="seq") as pbar:
            for i in range(0, total, batch_size):
                batch = self.sequences[i:i+batch_size]
                batch_vals = []
                for header, seq, label in batch:
                    vals, _ = self._extract_single(seq, return_names=False)
                    batch_vals.append(vals)
                all_features.extend(batch_vals)
                pbar.update(len(batch))

        df = pd.DataFrame(all_features, columns=feature_names)
        df.insert(0, 'label', [s[2] for s in self.sequences])
        df.insert(0, 'sequence_id', [s[0] for s in self.sequences])
        return df

    def _extract_single(self, seq: str, return_names: bool = False):
        if self.seq_type == 'prot':
            return self._protein_features(seq, return_names)
        else:
            return self._dna_features(seq, return_names)

    # ---------- Protein features ----------
    def _protein_features(self, seq: str, return_names: bool):
        features = []
        names = []
        seq_len = len(seq) if not return_names else 30

        # 1. AAC
        if return_names:
            for aa in AA_LIST:
                names.append(f'AAC_{AA_NAMES[aa]}')
        else:
            aa_counts = Counter(seq.upper())
            for aa in AA_LIST:
                features.append(aa_counts.get(aa, 0) / seq_len if seq_len else 0)

        # 2. Dipeptide composition
        if return_names:
            for aa1 in AA_LIST:
                for aa2 in AA_LIST:
                    names.append(f'DP_{aa1}{aa2}')
        else:
            dipep_counts = Counter(seq[i:i+2].upper() for i in range(seq_len-1))
            for aa1 in AA_LIST:
                for aa2 in AA_LIST:
                    dp = aa1+aa2
                    features.append(dipep_counts.get(dp, 0) / max(1, seq_len-1))

        # 3. Physicochemical
        phys_names = ['MW', 'Charge_pH7.4', 'Aromaticity', 'InstabilityIndex', 'GRAVY', 'IsoelectricPoint']
        if return_names:
            names.extend(phys_names)
        else:
            try:
                prot_param = ProtParam.ProteinAnalysis(seq)
                features.append(molecular_weight(seq, seq_type='protein'))
                features.append(prot_param.charge_at_pH(7.4))
                features.append(prot_param.aromaticity())
                features.append(prot_param.instability_index())
                features.append(prot_param.gravy())
                features.append(prot_param.isoelectric_point())
            except:
                features.extend([0.0]*6)

        # 4. Sequence order (lag 1..30)
        lag_count = 30
        if return_names:
            for lag in range(1, lag_count+1):
                names.append(f'SeqOrder_lag{lag}')
        else:
            hydrophobicity = self._get_hydrophobicity()
            for lag in range(1, min(lag_count+1, seq_len)):
                tau = 0
                cnt = 0
                for i in range(seq_len - lag):
                    a1 = seq[i].upper()
                    a2 = seq[i+lag].upper()
                    if a1 in hydrophobicity and a2 in hydrophobicity:
                        tau += (hydrophobicity[a1] - hydrophobicity[a2]) ** 2
                        cnt += 1
                features.append(tau / max(1, cnt))
            while len(features) < lag_count:
                features.append(0.0)

        # 5. PAAC (20 comp + 30 order)
        lambda_val = 30
        if return_names:
            for aa in AA_LIST:
                names.append(f'PAAC_comp_{AA_NAMES[aa]}')
            for lag in range(1, lambda_val+1):
                names.append(f'PAAC_order_lag{lag}')
        else:
            aa_counts = Counter(seq.upper())
            comp = [aa_counts.get(aa, 0) / seq_len for aa in AA_LIST]
            hydrophobicity = self._get_hydrophobicity()
            theta = []
            for lag in range(1, min(lambda_val+1, seq_len)):
                tau = 0
                cnt = 0
                for i in range(seq_len - lag):
                    a1 = seq[i].upper()
                    a2 = seq[i+lag].upper()
                    if a1 in hydrophobicity and a2 in hydrophobicity:
                        tau += (hydrophobicity[a1] - hydrophobicity[a2]) ** 2
                        cnt += 1
                theta.append(tau / max(1, cnt))
            while len(theta) < lambda_val:
                theta.append(0.0)
            w = 0.05
            denom = sum(comp) + w * sum(theta)
            if denom == 0:
                denom = 1
            for c in comp:
                features.append(c / denom)
            for t in theta:
                features.append(w * t / denom)

        # 6. CTD (simplified, 147 features)
        if return_names:
            prop_names = ['Hydrophobicity', 'Polarity', 'Charge']
            group_ids = ['G1','G2','G3']
            for pn in prop_names:
                for gid in group_ids:
                    names.append(f'CTD_Comp_{pn}_{gid}')
            trans_pairs = [('G1','G2'), ('G1','G3'), ('G2','G3')]
            for pn in prop_names:
                for p1,p2 in trans_pairs:
                    names.append(f'CTD_Trans_{pn}_{p1}_{p2}')
            for pn in prop_names:
                for gid in group_ids:
                    for pct in ['25%','50%','75%','100%']:
                        names.append(f'CTD_Dist_{pn}_{gid}_{pct}')
        else:
            groups = {
                'Hydrophobicity': {'G1': set('AGTS'), 'G2': set('CDENQHKR'), 'G3': set('ILMFWYV')},
                'Polarity': {'G1': set('AGTS'), 'G2': set('CDENQHKR'), 'G3': set('ILMFWYV')},
                'Charge': {'G1': set('KR'), 'G2': set('DE'), 'G3': set('ACGHILMNFQPSTVWY')}
            }
            for prop_name, prop_groups in groups.items():
                # Composition
                for gid in ['G1','G2','G3']:
                    gset = prop_groups[gid]
                    cnt = sum(1 for aa in seq.upper() if aa in gset)
                    features.append(cnt / seq_len)
                # Transition
                trans_pairs = [('G1','G2'), ('G1','G3'), ('G2','G3')]
                for g1,g2 in trans_pairs:
                    set1 = prop_groups[g1]
                    set2 = prop_groups[g2]
                    trans = 0
                    for i in range(seq_len-1):
                        a1 = seq[i].upper()
                        a2 = seq[i+1].upper()
                        if (a1 in set1 and a2 in set2) or (a1 in set2 and a2 in set1):
                            trans += 1
                    features.append(trans / max(1, seq_len-1))
                # Distribution
                for gid in ['G1','G2','G3']:
                    gset = prop_groups[gid]
                    pos = [i for i,aa in enumerate(seq.upper()) if aa in gset]
                    if pos:
                        total = len(pos)
                        for pct in [0.25, 0.50, 0.75, 1.0]:
                            idx = int(pct*total) - 1
                            if idx < 0:
                                idx = 0
                            features.append(pos[idx] / seq_len)
                    else:
                        features.extend([0.0]*4)

        # 7. Moran autocorrelation (30 lags)
        if return_names:
            for lag in range(1,31):
                names.append(f'Moran_lag{lag}')
        else:
            hydrophobicity = self._get_hydrophobicity()
            values = [hydrophobicity.get(aa.upper(), 0) for aa in seq]
            mean_val = np.mean(values)
            for lag in range(1, min(31, seq_len)):
                num = 0
                den = 0
                cnt = 0
                for i in range(seq_len - lag):
                    num += (values[i] - mean_val) * (values[i+lag] - mean_val)
                    cnt += 1
                for v in values:
                    den += (v - mean_val) ** 2
                if den == 0:
                    features.append(0.0)
                else:
                    features.append(num / (cnt * den))
            while len(features) < 30:
                features.append(0.0)

        # 8. BLOSUM62 encoding (20 values)
        if return_names:
            for aa in AA_LIST:
                names.append(f'BLOSUM_{aa}')
        else:
            for aa in AA_LIST:
                features.append(seq.upper().count(aa) / max(1, seq_len))

        if return_names:
            return None, names
        else:
            return features, None

    def _get_hydrophobicity(self):
        return {
            'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,
            'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
            'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,
            'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2
        }

    # ---------- DNA features ----------
    def _dna_features(self, seq: str, return_names: bool):
        features = []
        names = []
        seq = seq.upper()
        seq_len = len(seq) if not return_names else 30
        nucs = NUCS

        # 1. NAC
        if return_names:
            for n in nucs:
                names.append(f'NAC_{n}')
        else:
            nuc_counts = Counter(seq)
            for n in nucs:
                features.append(nuc_counts.get(n, 0) / seq_len if seq_len else 0)

        # 2. DNC
        if return_names:
            for n1 in nucs:
                for n2 in nucs:
                    names.append(f'DNC_{n1}{n2}')
        else:
            dinuc_counts = Counter(seq[i:i+2] for i in range(seq_len-1))
            for n1 in nucs:
                for n2 in nucs:
                    dn = n1+n2
                    features.append(dinuc_counts.get(dn, 0) / max(1, seq_len-1))

        # 3. TNC
        if return_names:
            for n1 in nucs:
                for n2 in nucs:
                    for n3 in nucs:
                        names.append(f'TNC_{n1}{n2}{n3}')
        else:
            trinuc_counts = Counter(seq[i:i+3] for i in range(seq_len-2))
            for n1 in nucs:
                for n2 in nucs:
                    for n3 in nucs:
                        tn = n1+n2+n3
                        features.append(trinuc_counts.get(tn, 0) / max(1, seq_len-2))

        # 4. K-mer (k=4)
        k = 4
        if return_names:
            import itertools
            for kmer in itertools.product(nucs, repeat=k):
                names.append(f'Kmer_{"".join(kmer)}')
        else:
            kmers = [''.join(p) for p in __import__('itertools').product(nucs, repeat=k)]
            kmer_counts = Counter(seq[i:i+k] for i in range(seq_len-k+1))
            for km in kmers:
                features.append(kmer_counts.get(km, 0) / max(1, seq_len-k+1))

        # 5. Pseudo dinucleotide composition (PseDNC)
        lambda_val = 10
        if return_names:
            for n1 in nucs:
                for n2 in nucs:
                    names.append(f'PseDNC_comp_{n1}{n2}')
            for lag in range(1, lambda_val+1):
                names.append(f'PseDNC_order_lag{lag}')
        else:
            dinuc_counts = Counter(seq[i:i+2] for i in range(seq_len-1))
            dinuc_comp = [dinuc_counts.get(n1+n2, 0) / max(1, seq_len-1)
                          for n1 in nucs for n2 in nucs]
            eiip = {'A':0.1260,'C':0.1340,'G':0.0806,'T':0.1335}
            theta = []
            for lag in range(1, min(lambda_val+1, seq_len)):
                tau = 0
                cnt = 0
                for i in range(seq_len - lag):
                    n1 = seq[i]
                    n2 = seq[i+lag]
                    if n1 in eiip and n2 in eiip:
                        tau += (eiip[n1] - eiip[n2]) ** 2
                        cnt += 1
                theta.append(tau / max(1, cnt))
            while len(theta) < lambda_val:
                theta.append(0.0)
            w = 0.05
            denom = sum(dinuc_comp) + w * sum(theta)
            if denom == 0:
                denom = 1
            for dc in dinuc_comp:
                features.append(dc / denom)
            for t in theta:
                features.append(w * t / denom)

        # 6. DNA Moran autocorrelation (30 lags)
        if return_names:
            for lag in range(1,31):
                names.append(f'DNA_Moran_lag{lag}')
        else:
            eiip = {'A':0.1260,'C':0.1340,'G':0.0806,'T':0.1335}
            values = [eiip.get(n, 0) for n in seq]
            mean_val = np.mean(values)
            for lag in range(1, min(31, seq_len)):
                num = 0
                den = 0
                cnt = 0
                for i in range(seq_len - lag):
                    num += (values[i] - mean_val) * (values[i+lag] - mean_val)
                    cnt += 1
                for v in values:
                    den += (v - mean_val) ** 2
                if den == 0:
                    features.append(0.0)
                else:
                    features.append(num / (cnt * den))
            while len(features) < 30:
                features.append(0.0)

        # 7. Enhanced NAC (ENAC) – 4 features
        if return_names:
            for n in nucs:
                names.append(f'ENAC_{n}')
        else:
            for n in nucs:
                weighted = 0
                for i, base in enumerate(seq):
                    if base == n:
                        weighted += (i+1) / seq_len
                features.append(weighted / seq_len)

        if return_names:
            return None, names
        else:
            return features, None


# ============================================================================
# FEATURE SELECTOR
# ============================================================================

class FeatureSelector:
    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.results = {}

    def select_all(self, n_selected: int = None, k_folds: int = 5) -> Dict:
        if n_selected is None:
            n_selected = min(100, self.X.shape[1] // 2)
        max_features = min(n_selected, self.X.shape[1])

        with tqdm(total=len(FS_METHODS), desc="Feature selection", unit="method") as pbar:
            for name, config in FS_METHODS.items():
                try:
                    result = self._run_method(name, config, max_features, k_folds)
                    self.results[name] = result
                except Exception as e:
                    print(f"  Warning: {name} failed: {e}")
                    self.results[name] = {'error': str(e), 'selected': [], 'scores': []}
                pbar.update(1)
        return self.results

    def _run_method(self, name, config, max_features, k_folds):
        X = self.X
        y = self.y

        if name == 'f_classif':
            selector = SelectKBest(f_classif, k=max_features)
            selector.fit(X, y)
            scores = selector.scores_
            indices = np.argsort(scores)[-max_features:][::-1]
        elif name == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=max_features)
            selector.fit(X, y)
            scores = selector.scores_
            indices = np.argsort(scores)[-max_features:][::-1]
        elif name == 'chi2':
            X_shifted = X - X.min(axis=0) + 1e-10
            selector = SelectKBest(chi2, k=max_features)
            selector.fit(X_shifted, y)
            scores = selector.scores_
            indices = np.argsort(scores)[-max_features:][::-1]
        elif name == 'rfe_rf':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=max_features)
            selector.fit(X, y)
            indices = np.where(selector.support_)[0]
            scores = selector.ranking_
        elif name == 'rf_importance':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator, max_features=max_features, threshold='median')
            selector.fit(X, y)
            indices = np.where(selector.get_support())[0]
            scores = selector.estimator_.feature_importances_
        elif name == 'lasso':
            estimator = LassoCV(cv=5, random_state=42)
            selector = SelectFromModel(estimator, max_features=max_features)
            selector.fit(X, y)
            indices = np.where(selector.get_support())[0]
            scores = np.abs(selector.estimator_.coef_)
        elif name == 'gb_importance':
            estimator = GradientBoostingClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator, max_features=max_features, threshold='median')
            selector.fit(X, y)
            indices = np.where(selector.get_support())[0]
            scores = selector.estimator_.feature_importances_
        else:
            raise ValueError(f"Unknown method: {name}")

        selected_features = [self.feature_names[i] for i in indices]
        performance = self._evaluate_selection(indices, k_folds)

        return {
            'selected_indices': indices.tolist(),
            'selected_features': selected_features,
            'scores': scores.tolist() if hasattr(scores, 'tolist') else scores,
            'performance': performance,
            'n_selected': len(indices),
        }

    def _evaluate_selection(self, indices, k_folds):
        X_sel = self.X[:, indices]
        y = self.y
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        skf = StratifiedKFold(n_splits=min(k_folds, len(np.unique(y))), shuffle=True, random_state=42)
        scores = cross_val_score(rf, X_sel, y, cv=skf, scoring='accuracy')
        all_metrics = {'accuracy_mean': np.mean(scores), 'accuracy_std': np.std(scores)}
        rf.fit(X_sel, y)
        y_pred = rf.predict(X_sel)
        all_metrics['precision'] = precision_score(y, y_pred, average='weighted', zero_division=0)
        all_metrics['recall'] = recall_score(y, y_pred, average='weighted', zero_division=0)
        all_metrics['f1'] = f1_score(y, y_pred, average='weighted', zero_division=0)
        try:
            all_metrics['mcc'] = matthews_corrcoef(y, y_pred)
        except:
            all_metrics['mcc'] = 0.0
        return all_metrics


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_lines = []

    def add_section(self, title: str, content: List[str] = None):
        self.report_lines.append("")
        self.report_lines.append("=" * 80)
        self.report_lines.append(f" {title}")
        self.report_lines.append("=" * 80)
        if content:
            self.report_lines.extend(content)

    def add_subsection(self, title: str, content: List[str] = None):
        self.report_lines.append("")
        self.report_lines.append("-" * 40)
        self.report_lines.append(f" {title}")
        self.report_lines.append("-" * 40)
        if content:
            self.report_lines.extend(content)

    def generate_sequence_stats(self, sequences, seq_type):
        seqs = [s[1] for s in sequences]
        labels = [s[2] for s in sequences]
        lengths = [len(s) for s in seqs]
        pos_seqs = [seqs[i] for i in range(len(seqs)) if labels[i] == 1]
        neg_seqs = [seqs[i] for i in range(len(seqs)) if labels[i] == 0]
        content = [
            f"Total sequences: {len(seqs)}",
            f"Positive (label=1): {len(pos_seqs)}",
            f"Negative (label=0): {len(neg_seqs)}",
            f"Sequence type: {seq_type.upper()}",
            "",
            "--- Sequence Length Statistics ---",
            f"Overall - Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.2f}, Std: {np.std(lengths):.2f}",
            f"Positive - Min: {min([len(s) for s in pos_seqs]) if pos_seqs else 0}, Max: {max([len(s) for s in pos_seqs]) if pos_seqs else 0}",
            f"Negative - Min: {min([len(s) for s in neg_seqs]) if neg_seqs else 0}, Max: {max([len(s) for s in neg_seqs]) if neg_seqs else 0}",
        ]
        if seq_type == 'nuc':
            all_chars = ''.join(seqs)
            comp = Counter(all_chars)
            content.append("")
            content.append("--- Nucleotide Composition ---")
            for n in 'ACGT':
                content.append(f"  {n}: {comp.get(n, 0) / len(all_chars) * 100:.2f}%")
        else:
            all_chars = ''.join(seqs)
            comp = Counter(all_chars)
            content.append("")
            content.append("--- Amino Acid Composition (Top 10) ---")
            for aa, count in comp.most_common(10):
                content.append(f"  {aa}: {count / len(all_chars) * 100:.2f}%")
        self.add_section("INPUT SEQUENCE STATISTICS", content)

    def generate_feature_stats(self, df: pd.DataFrame, name: str = "Consolidated"):
        feature_cols = [c for c in df.columns if c not in ['sequence_id', 'label']]
        n_features = len(feature_cols)
        n_samples = len(df)
        content = [
            f"Dataset: {name}",
            f"Number of samples: {n_samples}",
            f"Number of features: {n_features}",
            "",
            "--- Feature Value Statistics ---",
            f"Mean (across all features): {df[feature_cols].mean().mean():.4f}",
            f"Std (across all features): {df[feature_cols].std().mean():.4f}",
            f"Min (across all features): {df[feature_cols].min().min():.4f}",
            f"Max (across all features): {df[feature_cols].max().max():.4f}",
            "",
            "--- Per-Feature Statistics ---",
            f"Features with zero variance: {sum(df[feature_cols].var() == 0)}",
            f"Features with >90% missing: {sum(df[feature_cols].isnull().mean() > 0.9)}",
        ]
        self.add_section(f"FEATURE STATISTICS - {name}", content)

    def generate_selection_results(self, results: Dict, df: pd.DataFrame):
        content = ["Feature selection methods comparison:"]
        content.append("")
        content.append(f"{'Method':<20} {'Selected':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MCC':<12}")
        content.append("-" * 90)
        best_method = None
        best_acc = -1
        for name, result in results.items():
            if 'error' in result:
                content.append(f"{name:<20} {'ERROR':<10}")
                continue
            perf = result.get('performance', {})
            acc = perf.get('accuracy_mean', 0)
            content.append(
                f"{name:<20} {result['n_selected']:<10} {acc:.4f}     "
                f"{perf.get('precision', 0):.4f}     {perf.get('recall', 0):.4f}     "
                f"{perf.get('f1', 0):.4f}     {perf.get('mcc', 0):.4f}"
            )
            if acc > best_acc:
                best_acc = acc
                best_method = name
        self.add_section("FEATURE SELECTION RESULTS", content)

        for name, result in results.items():
            if 'error' in result:
                continue
            selected = result.get('selected_features', [])[:20]
            scores = result.get('scores', [])
            if scores and len(scores) == len(df.columns) - 2:
                score_map = {df.columns[i+2]: scores[i] for i in range(len(scores))}
                ranked = sorted(selected, key=lambda x: score_map.get(x, 0), reverse=True)
            else:
                ranked = selected
            content = [
                f"Method: {name.upper()}",
                f"Type: {FS_METHODS[name]['type']}",
                f"Features selected: {result['n_selected']}",
                f"Accuracy (CV): {result['performance']['accuracy_mean']:.4f} ± {result['performance']['accuracy_std']:.4f}",
                "",
                "Top 20 ranked features:",
            ]
            for i, f in enumerate(ranked[:20], 1):
                score = score_map.get(f, 0) if scores else 0
                content.append(f"  {i:3d}. {f} (score: {score:.4f})")
            self.add_subsection(f"Details: {name.upper()}", content)

    def generate_pipeline_summary(self, start_time: float, end_time: float, steps: Dict):
        content = [
            "Pipeline execution completed successfully.",
            "",
            f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}",
            f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total execution time: {end_time - start_time:.2f} seconds",
            "",
            "--- Step-wise execution times ---",
        ]
        for step, duration in steps.items():
            content.append(f"  {step}: {duration:.2f} seconds")
        self.add_section("PIPELINE EXECUTION SUMMARY", content)

    def save(self, filename: str = "report.txt"):
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            f.write("\n".join(self.report_lines))
        return report_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class BioLovePipeline:
    def __init__(self, seq_type: str, pos_file: str, neg_file: str,
                 output_dir: str, n_cores: int = 4):
        self.seq_type = seq_type
        self.pos_file = pos_file
        self.neg_file = neg_file
        self.output_dir = Path(output_dir)
        self.n_cores = n_cores
        self.start_time = time.time()
        self.step_times = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report = ReportGenerator(output_dir)
        self.sequences = []
        self.feature_df = None
        self.selection_results = None

    def run(self):
        print(f"\n{'='*60}")
        print(f" BioLove: Automated Feature Engineering Pipeline")
        print(f" Sequence Type: {self.seq_type.upper()}")
        print(f" Output Directory: {self.output_dir}")
        print(f" Cores: {self.n_cores}")
        print(f"{'='*60}\n")

        step_start = time.time()
        self._load_sequences()
        self.step_times['load_sequences'] = time.time() - step_start

        step_start = time.time()
        self.report.generate_sequence_stats(self.sequences, self.seq_type)
        self.step_times['sequence_stats'] = time.time() - step_start

        step_start = time.time()
        self._extract_features()
        self.step_times['feature_extraction'] = time.time() - step_start

        step_start = time.time()
        self.report.generate_feature_stats(self.feature_df, "Consolidated")
        self.step_times['feature_stats'] = time.time() - step_start

        step_start = time.time()
        self._feature_selection()
        self.step_times['feature_selection'] = time.time() - step_start

        step_start = time.time()
        self._save_outputs()
        self.step_times['save_outputs'] = time.time() - step_start

        step_start = time.time()
        self.report.generate_pipeline_summary(self.start_time, time.time(), self.step_times)
        report_path = self.report.save()
        self.step_times['final_report'] = time.time() - step_start

        print(f"\n{'='*60}")
        print(" PIPELINE COMPLETED SUCCESSFULLY")
        print(f" Total time: {time.time() - self.start_time:.2f} seconds")
        print(f" Report: {report_path}")
        print(f" Outputs: {self.output_dir}")
        print(f"{'='*60}\n")

    def _load_sequences(self):
        print("Loading sequences...")
        sequences = []
        for record in SeqIO.parse(self.pos_file, "fasta"):
            sequences.append((record.id, str(record.seq), 1))
        for record in SeqIO.parse(self.neg_file, "fasta"):
            sequences.append((record.id, str(record.seq), 0))
        if not sequences:
            raise ValueError("No sequences found in input files")
        self.sequences = sequences
        print(f"  Loaded {len(sequences)} sequences ({sum(1 for _,_,l in sequences if l==1)} positive, {sum(1 for _,_,l in sequences if l==0)} negative)")

    def _extract_features(self):
        print("\nExtracting features...")
        extractor = FeatureExtractor(self.seq_type, self.sequences)
        self.feature_df = extractor.extract_all()
        print(f"  Extracted {self.feature_df.shape[1]-2} features from {self.feature_df.shape[0]} sequences")
        consolidated_path = self.output_dir / "consolidated_features.csv"
        self.feature_df.to_csv(consolidated_path, index=False)
        print(f"  Saved consolidated features to {consolidated_path}")

    def _feature_selection(self):
        print("\nPerforming feature selection...")
        feature_cols = [c for c in self.feature_df.columns if c not in ['sequence_id', 'label']]
        X = self.feature_df[feature_cols].values
        y = self.feature_df['label'].values
        X = np.nan_to_num(X, nan=0.0)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        selector = FeatureSelector(X, y, feature_cols)
        self.selection_results = selector.select_all()
        self.report.generate_selection_results(self.selection_results, self.feature_df)

    def _save_outputs(self):
        print("\nSaving output datasets...")
        feature_cols = [c for c in self.feature_df.columns if c not in ['sequence_id', 'label']]
        for method_name, result in self.selection_results.items():
            if 'error' in result:
                continue
            selected_features = result.get('selected_features', [])
            if not selected_features:
                continue
            subset_cols = ['sequence_id', 'label'] + selected_features
            subset_df = self.feature_df[subset_cols]
            filename = f"selected_features_{method_name}.csv"
            subset_df.to_csv(self.output_dir / filename, index=False)
            self.report.generate_feature_stats(subset_df, f"Selected_{method_name.upper()}")

        # Ranking summary
        ranking_df = self._generate_ranking_summary()
        ranking_df.to_csv(self.output_dir / "feature_ranking_summary.csv", index=False)
        print(f"  Saved {len(self.selection_results)} feature subsets and ranking summary")

    def _generate_ranking_summary(self) -> pd.DataFrame:
        feature_cols = [c for c in self.feature_df.columns if c not in ['sequence_id', 'label']]
        rank_data = {'feature': feature_cols}
        for method_name, result in self.selection_results.items():
            if 'error' in result:
                continue
            selected = result.get('selected_features', [])
            scores = result.get('scores', [])
            if scores and len(scores) == len(feature_cols):
                rank_data[f'{method_name}_score'] = scores
                rank_data[f'{method_name}_rank'] = [len(feature_cols) - stats.rankdata(scores)[i] for i in range(len(scores))]
            else:
                rank_data[f'{method_name}_selected'] = [1 if f in selected else 0 for f in feature_cols]
        ranking_df = pd.DataFrame(rank_data)
        score_cols = [c for c in ranking_df.columns if c.endswith('_score')]
        if score_cols:
            for col in score_cols:
                max_val = ranking_df[col].max()
                min_val = ranking_df[col].min()
                if max_val > min_val:
                    ranking_df[f'{col}_norm'] = (ranking_df[col] - min_val) / (max_val - min_val)
                else:
                    ranking_df[f'{col}_norm'] = 0
            norm_cols = [c for c in ranking_df.columns if c.endswith('_norm')]
            ranking_df['aggregate_score'] = ranking_df[norm_cols].mean(axis=1)
            ranking_df = ranking_df.sort_values('aggregate_score', ascending=False)
        return ranking_df


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BioLove: Automated Feature Engineering and Selection Pipeline",
        epilog="""Examples:
  Protein: python3 biolove.py --type prot --pos positive.fasta --neg negative.fasta --cores 4 --out ./results
  DNA:     python3 biolove.py --type nuc --pos positive.fasta --neg negative.fasta --cores 4 --out ./results
        """
    )
    parser.add_argument('--type', required=True, choices=['prot', 'nuc'],
                        help='Sequence type: prot (protein) or nuc (DNA)')
    parser.add_argument('--pos', required=True,
                        help='Path to positive sequences FASTA file')
    parser.add_argument('--neg', required=True,
                        help='Path to negative sequences FASTA file')
    parser.add_argument('--cores', type=int, default=4,
                        help='Number of CPU cores for parallel processing (default: 4)')
    parser.add_argument('--out', required=True,
                        help='Output directory path')

    args = parser.parse_args()

    if not os.path.exists(args.pos):
        print(f"Error: Positive file not found: {args.pos}")
        sys.exit(1)
    if not os.path.exists(args.neg):
        print(f"Error: Negative file not found: {args.neg}")
        sys.exit(1)

    pipeline = BioLovePipeline(
        seq_type=args.type,
        pos_file=args.pos,
        neg_file=args.neg,
        output_dir=args.out,
        n_cores=args.cores
    )

    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
