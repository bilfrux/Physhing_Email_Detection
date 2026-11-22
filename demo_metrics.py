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

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from data_profiler import DatasetProfiler
from metrics_evaluation import PhishingModelEvaluator, plot_comparison_chart


def load_project_data() -> pd.DataFrame:
    """
    Charge le corpus dédoublonné et le split figé produits par le pipeline
    (scripts/01_audit_and_build.py et scripts/02_split.py).

    Corrections (AI_AUDIT #11) : l'ancienne version faisait un train_test_split aléatoire
    sur un CSV brut non dédoublonné (fuite des quasi-doublons entre train et test),
    transformait les labels invalides en 0 sans prévenir, et générait un corpus simulé
    si le CSV manquait. On s'arrête maintenant avec une erreur au lieu de simuler.
    """
    root = Path(__file__).resolve().parent
    corpus, split = root / "data/processed/corpus.parquet", root / "data/processed/split.csv"
    if not (corpus.exists() and split.exists()):
        raise SystemExit("data/processed/ introuvable : lancer d'abord bash run_all.sh (étapes 01 et 02).")
    df = pd.read_parquet(corpus, columns=["source", "label", "text_min"]).rename(columns={"text_min": "text"})
    df["split"] = pd.read_csv(split)["split"].values
    assert df["label"].isin([0, 1]).all(), "labels must be 0/1"
    return df


def main():
    print("=" * 70)
    print("🎯 WORKFLOW GLOBAL : PROFILAGE, BIAIS & BENCHMARK D'ÉVALUATION CYBER-ML")
    print("=" * 70)

    # ---------------------------------------------------------
    # ÉTAPE 1 : Chargement et Profilage des Données
    # ---------------------------------------------------------
    df = load_project_data()
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]

    profiler = DatasetProfiler(
        df=df[["text", "label"]],
        text_col="text",
        label_col="label",
        dataset_name="Corpus dédoublonné (6 sources)"
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
    # Split figé du projet (dédoublonné, groupé, stratifié source x label) : même test set que la baseline.
    X_train_text, X_test_text = train["text"], test["text"]
    y_train, y_test = train["label"], test["label"]

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
    # (Supprimé : une "attaque adversariale" qui multipliait à la main les probabilités de 35 % des
    # phishing par 0.3. Ce n'était pas une attaque mais un résultat inventé ; AI_AUDIT #11.)

    evaluator_lr = PhishingModelEvaluator(
        y_true=y_test, y_proba=y_proba_lr, threshold=0.50, model_name="Baseline Reg. Logistique"
    )

    benchmark_results = [
        evaluator_lr.compute_all_metrics(target_tpr=0.95),
        evaluator_svm.compute_all_metrics(target_tpr=0.95),
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

    print("\n📋 TABLEAU COMPARATIF (test set figé du projet) :")
    print(summary_df.to_string(index=False))

    print("\n✅ Tous les graphiques et rapports ont été générés avec succès :")
    print("   1. data_profiling_and_bias.png         (Audit statistique, distributions & biais)")
    print("   2. model_decision_radar.png            (Radar décisionnel pour choix de modèles)")
    print("   3. phishing_evaluation_dashboard.png   (Matrice confusion, ROC, PR, Seuil)")
    print("   4. model_comparison_benchmark.png      (Benchmark Baseline vs SVM)")


if __name__ == "__main__":
    main()
