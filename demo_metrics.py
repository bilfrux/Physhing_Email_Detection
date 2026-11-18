"""
demo_metrics.py
---------------
Script de démonstration de bout en bout :
1. Profilage des données & Détection de biais (déséquilibre, doublons, fuites, heuristiques cyber).
2. Aide à la décision automatisée : Quelle métrique choisir ? Quel modèle privilégier (Linéaire, SVM, Arbres, Transformers) ?
3. Visualisations haute résolution :
   - data_profiling_and_bias.png
   - model_decision_radar.png
   - phishing_evaluation_dashboard.png
   - model_comparison_benchmark.png
"""

import os
import glob
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from data_profiler import DatasetProfiler
from metrics_evaluation import PhishingModelEvaluator, plot_comparison_chart


def load_or_generate_dataset(preferred_corpus: str = "SpamAssasin") -> Tuple[pd.DataFrame, str]:
    """
    Charge l'un des datasets réels du projet s'il existe (ex. SpamAssasin.csv, CEAS_08.csv),
    ou génère un dataset de simulation réaliste avec déséquilibre et patterns cyber.
    """
    candidate_paths = [
        f"phishing-email-dataset/versions/1/{preferred_corpus}.csv",
        "phishing-email-dataset/versions/1/SpamAssasin.csv",
        "phishing-email-dataset/versions/1/Enron.csv",
        "phishing-email-dataset/versions/1/CEAS_08.csv"
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            print(f"📂 Chargement du dataset réel : {path}...")
            # Lecture des colonnes utiles du dataset complet
            cols_to_read = None
            sample_df = pd.read_csv(path, nrows=2)
            if "label" in sample_df.columns:
                target_cols = [c for c in ["subject", "body", "text_combined", "label"] if c in sample_df.columns]
                df = pd.read_csv(path, usecols=target_cols)
            else:
                df = pd.read_csv(path)

            text_parts = []
            if "subject" in df.columns:
                text_parts.append(df["subject"].fillna(""))
            if "body" in df.columns:
                text_parts.append(df["body"].fillna(""))
            elif "text_combined" in df.columns:
                text_parts.append(df["text_combined"].fillna(""))

            df["text"] = text_parts[0] if len(text_parts) == 1 else (text_parts[0] + " " + text_parts[1])
            df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

            # S'assurer que les 2 classes sont bien présentes
            if df["label"].nunique() >= 2:
                name = os.path.basename(path).replace(".csv", "")
                return df[["text", "label"]], f"Dataset Réel ({name})"

    print("⚠️ Aucun fichier CSV réel trouvé localement. Génération d'un corpus de simulation réaliste...")
    np.random.seed(42)
    n_samples = 3000
    prevalence = 0.18  # 18% de phishing, 82% de légitimes (déséquilibre réaliste)

    labels = (np.random.rand(n_samples) < prevalence).astype(int)

    legit_templates = [
        "Hi team, please find attached the meeting minutes and project update for this week.",
        "Attached is the monthly financial report for your review. Let me know if you have questions.",
        "Reminder: All hands meeting scheduled tomorrow at 10 AM in room 4B.",
        "Can you send over the updated spreadsheet before end of day? Thanks.",
        "Lunch today at the cafeteria? Let me know."
    ]

    phish_templates = [
        "URGENT: Your account access has been suspended! Verify your password immediately: https://secure-bank-login.com",
        "Security Alert: Unauthorized login attempt detected from Russia. Click here to confirm identity: http://verify-auth.xyz",
        "Action Required: Your mailbox is full and will expire in 24 hours. Update your credentials at www.webmail-service.net",
        "FINAL NOTICE: Wire transfer pending. Immediate action needed to avoid permanent suspension: http://urgent-security.cc",
        "Dear customer, suspicious transaction on your account. Access your statement now: https://paypal-security-auth.net"
    ]

    texts = []
    for lbl in labels:
        if lbl == 1:
            base = np.random.choice(phish_templates)
            # Ajout d'insistance et de bruit
            if np.random.rand() > 0.5:
                base = base.upper()
        else:
            base = np.random.choice(legit_templates)
            if np.random.rand() > 0.7:
                base += " Best regards, Corporate Team."
        texts.append(base)

    # Introduire artificiellement quelques doublons (phénomène très fréquent dans les spams)
    duplicate_indices = np.random.choice(range(n_samples), size=int(n_samples * 0.06), replace=False)
    for idx in duplicate_indices:
        texts[idx] = texts[max(0, idx - 1)]

    df = pd.DataFrame({"text": texts, "label": labels})
    return df, "Corpus Simulé (Phishing Déséquilibré)"


def main():
    print("=" * 70)
    print("🎯 WORKFLOW GLOBAL : PROFILAGE, BIAIS & BENCHMARK D'ÉVALUATION CYBER-ML")
    print("=" * 70)

    # ---------------------------------------------------------
    # ÉTAPE 1 : Chargement et Profilage des Données
    # ---------------------------------------------------------
    df, dataset_name = load_or_generate_dataset()

    profiler = DatasetProfiler(
        df=df,
        text_col="text",
        label_col="label",
        dataset_name=dataset_name
    )

    # Affichage du rapport complet d'analyse de données et d'aide à la décision
    profiler.print_profiling_report()

    # Sauvegarde des visualisations de profilage et du guide de décision
    print("\n📊 Génération des graphiques exploratoires...")
    profiler.plot_data_profile_dashboard(save_path="data_profiling_and_bias.png")
    profiler.plot_model_recommendation_radar(save_path="model_decision_radar.png")

    # ---------------------------------------------------------
    # ÉTAPE 2 : Entraînement de Modèles Concrets (Baseline & SVM)
    # ---------------------------------------------------------
    print("\n⚙️ Entraînement et Évaluation des Modèles...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train_text)
    X_test_vec = tfidf.transform(X_test_text)

    # Modèle 1 : Baseline Régression Logistique (Al-Subaiey et al., 2024)
    model_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model_lr.fit(X_train_vec, y_train)
    y_proba_lr = model_lr.predict_proba(X_test_vec)[:, 1]

    # Modèle 2 : Support Vector Machine (LinearSVC calibré avec probabilités)
    model_svm_base = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    model_svm = CalibratedClassifierCV(estimator=model_svm_base, cv=3)
    model_svm.fit(X_train_vec, y_train)
    y_proba_svm = model_svm.predict_proba(X_test_vec)[:, 1]

    # ---------------------------------------------------------
    # ÉTAPE 3 : Évaluation Détaillée du Modèle Principal (SVM)
    # ---------------------------------------------------------
    evaluator_svm = PhishingModelEvaluator(
        y_true=y_test,
        y_proba=y_proba_svm,
        threshold=0.50,
        model_name="Linear SVM (Calibré)"
    )

    print("\n🔍 Rapport d'évaluation détaillé du SVM :")
    evaluator_svm.print_summary(target_tpr=0.95)

    evaluator_svm.plot_full_dashboard(
        target_tpr=0.95,
        save_path="phishing_evaluation_dashboard.png"
    )

    # ---------------------------------------------------------
    # ÉTAPE 4 : Benchmark Comparatif & Robustesse Cyber
    # ---------------------------------------------------------
    # Modèle 3 : Simulation d'une Attaque Adversariale d'Évasion (ex. salage ou substitution de tokens)
    y_proba_attacked = y_proba_svm.copy()
    phish_test_idx = np.where(y_test.values == 1)[0]
    attacked_phish = np.random.choice(phish_test_idx, size=int(len(phish_test_idx) * 0.35), replace=False)
    y_proba_attacked[attacked_phish] *= 0.30  # L'attaque trompe le classifieur

    evaluator_lr = PhishingModelEvaluator(
        y_true=y_test, y_proba=y_proba_lr, threshold=0.50, model_name="Baseline Reg. Logistique"
    )
    evaluator_attack = PhishingModelEvaluator(
        y_true=y_test, y_proba=y_proba_attacked, threshold=0.50, model_name="SVM sous Attaque (Evasion)"
    )

    benchmark_results = [
        evaluator_lr.compute_all_metrics(target_tpr=0.95),
        evaluator_svm.compute_all_metrics(target_tpr=0.95),
        evaluator_attack.compute_all_metrics(target_tpr=0.95)
    ]

    plot_comparison_chart(
        evaluation_results=benchmark_results,
        save_path="model_comparison_benchmark.png"
    )

    summary_df = pd.DataFrame(benchmark_results)[[
        "Model Name", "Precision", "Recall (TPR)", "F1-Score",
        "F2-Score (Recall-focused)", "MCC", "PR-AUC (Average Precision)",
        "ROC-AUC", "FPR @ 95% Recall"
    ]]

    print("\n📋 TABLEAU COMPARATIF MULTI-MODÈLES & SOUS ATTAQUE :")
    print(summary_df.to_string(index=False))

    print("\n✅ Tous les graphiques et rapports ont été générés avec succès :")
    print("   1. data_profiling_and_bias.png         (Audit statistique, distributions & biais)")
    print("   2. model_decision_radar.png            (Radar décisionnel pour choix de modèles)")
    print("   3. phishing_evaluation_dashboard.png   (Matrice confusion, ROC, PR, Seuil)")
    print("   4. model_comparison_benchmark.png      (Benchmark Baseline vs SVM vs Attaque)")


if __name__ == "__main__":
    main()
