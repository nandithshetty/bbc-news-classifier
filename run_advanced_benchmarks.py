import pandas as pd
import re
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (VotingClassifier, RandomForestClassifier,
                               AdaBoostClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import fetch_20newsgroups
import xgboost as xgb

print("=" * 60)
print("ADVANCED BENCHMARK: Multi-Dataset NLP Classification")
print("=" * 60)

# ------------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

def load_bbc():
    df = pd.read_csv("data/bbc_news.csv")
    df['Text'] = df['Text'].apply(clean_text)
    le = LabelEncoder()
    y = le.fit_transform(df['Category'])
    return df['Text'].tolist(), y, le.classes_, "BBC News"

def load_20newsgroups():
    cats = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    data = fetch_20newsgroups(subset='all', categories=cats,
                              remove=('headers', 'footers', 'quotes'))
    X = [clean_text(t) for t in data.data]
    y = data.target
    return X, y, data.target_names, "20 Newsgroups (4-class)"

def load_imdb():
    df = pd.read_csv("data/IMDB-Dataset.csv")
    df = df.dropna(subset=['review', 'sentiment'])
    df_pos = df[df['sentiment'].str.lower() == 'positive'].sample(n=1000, random_state=42)
    df_neg = df[df['sentiment'].str.lower() == 'negative'].sample(n=1000, random_state=42)
    df_sampled = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    X = [clean_text(text) for text in df_sampled['review']]
    y = np.array([1 if s.lower() == 'positive' else 0 for s in df_sampled['sentiment']])
    return X, y, ['negative', 'positive'], "IMDb Movie Reviews (Sentiment)"

datasets = [load_bbc(), load_20newsgroups(), load_imdb()]

# ------------------------------------------------------------------
# 2. MODEL DEFINITIONS
# ------------------------------------------------------------------
def get_models():
    lr   = LogisticRegression(max_iter=1000, random_state=42)
    nb   = MultinomialNB()
    svm  = SVC(kernel='rbf', probability=True, random_state=42, cache_size=1000, tol=0.01)
    lsvm = CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42, tol=0.01))

    ensemble_full = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('nb', MultinomialNB()),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42, cache_size=1000, tol=0.01))
        ], voting='soft')

    # Ablation variants
    abl_lr_nb = VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('nb', MultinomialNB())
    ], voting='soft')

    abl_lr_svm = VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42, cache_size=1000, tol=0.01))
    ], voting='soft')

    abl_nb_svm = VotingClassifier(estimators=[
        ('nb', MultinomialNB()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42, cache_size=1000, tol=0.01))
    ], voting='soft')

    rf      = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    ada     = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME')
    xgboost = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False,
                                  eval_metric='mlogloss', random_state=42,
                                  verbosity=0, n_jobs=-1)

    return {
        # Main baselines
        'LR':               LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes':      MultinomialNB(),
        'RBF SVM':          SVC(kernel='rbf', probability=True, random_state=42, cache_size=1000, tol=0.01),
        'Linear SVM':       lsvm,
        'Random Forest':    rf,
        'AdaBoost':         ada,
        'XGBoost':          xgboost,
        # Full ensemble
        'Ensemble (Full)':  ensemble_full,
        # Ablation
        'Abl: LR+NB':       abl_lr_nb,
        'Abl: LR+SVM':      abl_lr_svm,
        'Abl: NB+SVM':      abl_nb_svm,
    }

# ------------------------------------------------------------------
# 3. CROSS-VALIDATION RUNNER
# ------------------------------------------------------------------
def run_cv(X, y, model_name, model, n_splits=5):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    train_times, infer_times = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr = [X[i] for i in train_idx]
        X_te = [X[i] for i in test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]

        vec = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
        Xtr = vec.fit_transform(X_tr)
        Xte = vec.transform(X_te)

        t0 = time.perf_counter()
        model.fit(Xtr, y_tr)
        train_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        y_pred = model.predict(Xte)
        infer_times.append((time.perf_counter() - t0) / len(y_te) * 1000)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average='macro'))

    return {
        'acc_mean':  np.mean(accs),
        'acc_std':   np.std(accs),
        'f1_mean':   np.mean(f1s),
        'f1_std':    np.std(f1s),
        'acc_folds': accs,
        'train_time': np.mean(train_times),
        'infer_ms':   np.mean(infer_times),
    }

# ------------------------------------------------------------------
# 4. RUN ALL EXPERIMENTS
# ------------------------------------------------------------------
import os
import json

cache_path = "data/benchmark_results.json"
if os.path.exists(cache_path):
    with open(cache_path, "r") as f:
        all_results = json.load(f)
    print(f"Loaded existing results from cache: {cache_path}")
else:
    all_results = {}

for X, y, classes, ds_name in datasets:
    print(f"\n>>> Dataset: {ds_name}  ({len(X)} samples, {len(classes)} classes)")
    models = get_models()
    if ds_name not in all_results:
        all_results[ds_name] = {}

    for mname, model in models.items():
        if mname in all_results[ds_name]:
            res = all_results[ds_name][mname]
            print(f"    Training: {mname} ... (LOADED FROM CACHE) Acc={res['acc_mean']*100:.2f}% ± {res['acc_std']*100:.2f}%")
            continue
            
        print(f"    Training: {mname} ...", end='', flush=True)
        res = run_cv(list(X), np.array(y), mname, model)
        all_results[ds_name][mname] = res
        print(f" Acc={res['acc_mean']*100:.2f}% ± {res['acc_std']*100:.2f}%")
        
        # Save to cache file after each training step
        os.makedirs("data", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(all_results, f, indent=4)

# ------------------------------------------------------------------
# 5. STATISTICAL T-TESTS ON ABLATION (BBC News only)
# ------------------------------------------------------------------
print("\n\n--- Ablation Statistical Tests (BBC News) ---")
bbc_res = all_results["BBC News"]
full_folds = bbc_res['Ensemble (Full)']['acc_folds']

ablation_models = ['Abl: LR+NB', 'Abl: LR+SVM', 'Abl: NB+SVM']
print(f"\n{'Comparison':<30} {'p-value':>10} {'Significant?':>14}")
print("-" * 58)
for abl in ablation_models:
    abl_folds = bbc_res[abl]['acc_folds']
    _, pval = stats.ttest_rel(full_folds, abl_folds)
    sig = "YES (p<0.05)" if pval < 0.05 else "No"
    print(f"{'Full vs ' + abl:<30} {pval:>10.4f} {sig:>14}")

# Also ensemble vs Linear SVM
_, pval_lsvm = stats.ttest_rel(
    full_folds, bbc_res['Linear SVM']['acc_folds'])
print(f"{'Full Ensemble vs Linear SVM':<30} {pval_lsvm:>10.4f} "
      f"{'YES (p<0.05)' if pval_lsvm < 0.05 else 'No':>14}")

# ------------------------------------------------------------------
# 6. PRINT MARKDOWN TABLES
# ------------------------------------------------------------------
main_models = ['LR', 'Naive Bayes', 'RBF SVM', 'Linear SVM',
               'Random Forest', 'AdaBoost', 'XGBoost', 'Ensemble (Full)']

print("\n\n--- TABLE II: Benchmark Results (Accuracy % ± Std) ---")
header = f"{'Model':<22}" + "".join(f"  {ds_name[:15]:<15}" for *_, ds_name in datasets)
print(header)
print("-" * len(header))
for m in main_models:
    row = f"{m:<22}"
    for _, _, _, ds_name in datasets:
        r = all_results[ds_name][m]
        row += f"  {r['acc_mean']*100:.2f}±{r['acc_std']*100:.2f}    "
    print(row)

print("\n\n--- TABLE III: Ablation Study (BBC News) ---")
abl_names = ['Ensemble (Full)', 'Abl: LR+NB', 'Abl: LR+SVM', 'Abl: NB+SVM']
print(f"{'Model':<24} {'Accuracy':>10}  {'Delta vs Full':>13}")

base_acc = bbc_res['Ensemble (Full)']['acc_mean'] * 100
for m in abl_names:
    acc = bbc_res[m]['acc_mean'] * 100
    delta = acc - base_acc
    print(f"{m:<24} {acc:>10.2f}%  {delta:>+10.2f}%")

print("\n\n--- TABLE IV: Computational Cost (BBC News) ---")
print(f"{'Model':<22} {'Train Time (s)':>16} {'Infer (ms/sample)':>20}")
print("-" * 62)
for m in main_models:
    r = bbc_res[m]
    print(f"{m:<22} {r['train_time']:>16.3f} {r['infer_ms']:>20.4f}")

# ------------------------------------------------------------------
# 7. GENERATE PUBLICATION FIGURES
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12,
                     'figure.titlesize': 13, 'font.family': 'sans-serif'})

# Fig A: Multi-dataset grouped bar chart (main models only)
ds_names  = [ds for _, _, _, ds in datasets]
x         = np.arange(len(ds_names))
bar_width  = 0.1
n_models   = len(main_models)
colors     = plt.cm.tab10(np.linspace(0, 1, n_models))

fig, ax = plt.subplots(figsize=(13, 6))
for i, m in enumerate(main_models):
    means = [all_results[ds][m]['acc_mean'] * 100 for ds in ds_names]
    stds  = [all_results[ds][m]['acc_std']  * 100 for ds in ds_names]
    offset = (i - n_models / 2 + 0.5) * bar_width
    ax.bar(x + offset, means, bar_width, yerr=stds,
           label=m, capsize=3, color=colors[i], alpha=0.88)

ax.set_ylabel('Mean Accuracy (%) with ±1σ error bars')
ax.set_title('Multi-Dataset Performance Comparison (Stratified 5-Fold CV)')
ax.set_xticks(x)
ax.set_xticklabels(ds_names, fontsize=11)
ax.set_ylim(50, 105)
ax.legend(loc='lower right', fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig('advanced_comparison.png', dpi=300)
plt.close()
print("\nSaved: advanced_comparison.png")

# Fig B: Ablation study bar chart
abl_labels = ['Full\n(LR+NB+SVM)', 'Ablation\n(LR+NB)', 'Ablation\n(LR+SVM)', 'Ablation\n(NB+SVM)']
abl_accs   = [bbc_res[m]['acc_mean'] * 100 for m in abl_names]
abl_stds   = [bbc_res[m]['acc_std']  * 100 for m in abl_names]
bar_colors = ['#4A90E2', '#E87C3E', '#E87C3E', '#E87C3E']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(abl_labels, abl_accs, yerr=abl_stds, capsize=5,
              color=bar_colors, alpha=0.88, edgecolor='none')
ax.axhline(y=abl_accs[0], color='#4A90E2', linestyle='--', linewidth=1.2,
           label='Full Ensemble Baseline')
ax.set_ylabel('Mean Accuracy (%) with ±1σ', fontsize=11)
ax.set_title('Ablation Study — Soft-Voting Ensemble (BBC News, 5-Fold CV)', fontsize=12)
ax.set_ylim(min(abl_accs) - 1.5, max(abl_accs) + 1.5)
ax.legend(fontsize=9)
for bar, val, std in zip(bars, abl_accs, abl_stds):
    ax.text(bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + std + 0.15,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('ablation_study.png', dpi=300)
plt.close()
print("Saved: ablation_study.png")

# Fig C: Computational efficiency (training time bar chart)
eff_models = main_models
train_ts  = [bbc_res[m]['train_time'] for m in eff_models]
infer_ms  = [bbc_res[m]['infer_ms']   for m in eff_models]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.barh(eff_models, train_ts, color='#4A90E2', alpha=0.85)
ax1.set_xlabel('Mean Training Time (s)')
ax1.set_title('Training Time per Fold')
ax1.invert_yaxis()

ax2.barh(eff_models, infer_ms, color='#50E3C2', alpha=0.85)
ax2.set_xlabel('Mean Inference Latency (ms/sample)')
ax2.set_title('Inference Latency per Sample')
ax2.invert_yaxis()

plt.suptitle('Computational Efficiency Analysis (BBC News)', fontsize=13)
plt.tight_layout()
plt.savefig('computational_efficiency.png', dpi=300)
plt.close()
print("Saved: computational_efficiency.png")

print("\n\nAll experiments complete!")
