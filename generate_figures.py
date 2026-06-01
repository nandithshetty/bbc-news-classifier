import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from scipy import stats

# Set styling for academic paper
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Load dataset
df = pd.read_csv("d:/Project/ML/data/bbc_news.csv")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

df['Text'] = df['Text'].apply(clean_text)

# Encode labels
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])
target_names = le.classes_

X = df['Text']
y = df['Category']

# Stratified 5-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Multinomial NB': MultinomialNB(),
    'Linear SVM': SVC(kernel='linear', probability=True, random_state=42),
    'RBF SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Ensemble (LR+NB+RBF)': VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('nb', MultinomialNB()),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42))
        ],
        voting='soft'
    )
}

# Store metrics and confusion matrices
results = {name: {'accuracy': [], 'f1_macro': []} for name in models}
aggregated_cm = np.zeros((len(target_names), len(target_names)))

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"Processing Fold {fold + 1}/5...")
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results[name]['accuracy'].append(acc)
        results[name]['f1_macro'].append(f1)
        
        # Accumulate confusion matrix for the Ensemble
        if name == 'Ensemble (LR+NB+RBF)':
            aggregated_cm += confusion_matrix(y_test, y_pred)

print("\nAll folds complete. Conducting statistical testing...")

# Statistical Significance (Paired T-Tests between Ensemble and constituent models)
ensemble_accs = results['Ensemble (LR+NB+RBF)']['accuracy']
p_values = {}
for name in ['Logistic Regression', 'Multinomial NB', 'RBF SVM']:
    other_accs = results[name]['accuracy']
    t_stat, p_val = stats.ttest_rel(ensemble_accs, other_accs)
    p_values[name] = p_val
    print(f"Paired t-test (Ensemble vs {name}): p-value = {p_val:.6f}")

# 1. Generate Model Comparison Plot with Error Bars (Accuracy & F1)
labels = list(models.keys())
acc_means = [np.mean(results[name]['accuracy']) for name in models]
acc_stds = [np.std(results[name]['accuracy']) for name in models]
f1_means = [np.mean(results[name]['f1_macro']) for name in models]
f1_stds = [np.std(results[name]['f1_macro']) for name in models]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, acc_means, width, yerr=acc_stds, label='Accuracy', capsize=5, color='#4A90E2', edgecolor='none', alpha=0.9)
rects2 = ax.bar(x + width/2, f1_means, width, yerr=f1_stds, label='Macro F1-Score', capsize=5, color='#50E3C2', edgecolor='none', alpha=0.9)

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison with 5-Fold Cross-Validation Standard Deviation')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha='right')
ax.set_ylim(0.90, 1.00)  # Zoom in to see differences clearly
ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig("d:/Project/ML/model_comparison.png", dpi=300)
plt.close()
print("Saved model_comparison.png")

# 2. Generate Beautiful Aggregated Confusion Matrix Heatmap for Ensemble
plt.subplots(figsize=(8, 6))
# Normalize by row (recall)
cm_normalized = aggregated_cm.astype('float') / aggregated_cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Proportion of Correct Predictions'})
plt.title('Aggregated Confusion Matrix (Stratified 5-Fold CV) - Ensemble Model')
plt.ylabel('True Category')
plt.xlabel('Predicted Category')
plt.tight_layout()
plt.savefig("d:/Project/ML/confusion_matrix.png", dpi=300)
plt.close()
print("Saved confusion_matrix.png")

# Print summaries for tables
print("\n--- Summary Table for LaTeX/Markdown ---")
print("| Model | Mean Accuracy | Mean Macro F1 |")
print("|---|---|---|")
for name in models:
    print(f"| {name} | {np.mean(results[name]['accuracy']):.4f} ± {np.std(results[name]['accuracy']):.4f} | {np.mean(results[name]['f1_macro']):.4f} ± {np.std(results[name]['f1_macro']):.4f} |")
