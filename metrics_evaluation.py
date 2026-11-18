"""
metrics_evaluation.py
---------------------
Module d'évaluation et de visualisation pour la détection d'e-mails de phishing en Machine Learning.

Métriques et visualisations implémentées :
1. Matrice de Confusion annotée (TN, FP, FN, TP) avec focus cyber (coût FN vs FP).
2. Trio standard : Précision, Rappel (Recall / TPR), F1-score.
3. ROC Curve & ROC-AUC (avec mise en garde sur le déséquilibre de classes).
4. PR Curve (Precision-Recall) & PR-AUC / Average Precision (métrique reine en détection d'intrusion).
5. F-beta scores (F2 pénalisant les faux négatifs, F0.5 pénalisant les fausses alertes).
6. FPR à TPR fixé (ex. FPR @ 95% Recall, métrique de référence USENIX/IEEE S&P/CCS).
7. MCC (Matthews Correlation Coefficient, robuste aux variations de distribution en cross-corpus).
8. Dashboard d'analyse de seuil de décision (Threshold Tuning).
9. Visualisation comparative multi-modèles / cross-corpus / sous attaque.
"""

from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef
)


def compute_fpr_at_fixed_tpr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_tpr: float = 0.95
) -> Tuple[float, float]:
    """
    Calcule le FPR (False Positive Rate) pour un TPR (True Positive Rate / Recall) cible.
    Métrique standard des conférences de sécurité (IEEE S&P, USENIX Security, ACM CCS).

    Args:
        y_true: Labels binaires réels (0 = légitime, 1 = phishing).
        y_proba: Probabilités prédites pour la classe positive (phishing).
        target_tpr: Taux de détection souhaité (ex: 0.95 pour 95% de recall).

    Returns:
        (best_fpr, chosen_threshold) : Le FPR obtenu et le seuil de coupure correspondant.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    # Trouver l'indice où tpr >= target_tpr avec le plus faible FPR
    eligible_indices = np.where(tpr >= target_tpr)[0]
    if len(eligible_indices) == 0:
        return 1.0, 0.0

    idx = eligible_indices[np.argmin(fpr[eligible_indices])]
    best_fpr = float(fpr[idx])
    chosen_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.0
    return best_fpr, chosen_threshold


class PhishingModelEvaluator:
    """
    Gestionnaire complet d'évaluation et de visualisation de modèles de détection de phishing.
    """

    def __init__(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_proba: Union[np.ndarray, List[float]],
        threshold: float = 0.5,
        model_name: str = "Phishing Detector"
    ):
        """
        Initialise l'évaluateur.

        Args:
            y_true: Labels réels (0 = légitime/ham, 1 = phishing/spam malveillant).
            y_proba: Probabilités d'appartenance à la classe phishing (classe 1).
            threshold: Seuil de décision par défaut (défaut = 0.5).
            model_name: Nom du modèle pour les affichages et graphiques.
        """
        self.y_true = np.asarray(y_true).astype(int)
        self.y_proba = np.asarray(y_proba).astype(float)
        self.threshold = threshold
        self.model_name = model_name

        # Prédictions binaires basées sur le seuil
        self.y_pred = (self.y_proba >= self.threshold).astype(int)

    def compute_all_metrics(
        self,
        target_tpr: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calcule toutes les métriques indispensables recommandées pour le projet.

        Returns:
            Dictionnaire contenant toutes les métriques numériques et le diagnostic cyber.
        """
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        total = len(self.y_true)
        positives = tp + fn
        negatives = tn + fp

        # Standard metrics
        acc = accuracy_score(self.y_true, self.y_pred)
        prec = precision_score(self.y_true, self.y_pred, zero_division=0)
        rec = recall_score(self.y_true, self.y_pred, zero_division=0)  # TPR
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)
        f2 = fbeta_score(self.y_true, self.y_pred, beta=2.0, zero_division=0)
        f05 = fbeta_score(self.y_true, self.y_pred, beta=0.5, zero_division=0)
        mcc = matthews_corrcoef(self.y_true, self.y_pred)

        # Courbes et AUCs
        fpr_arr, tpr_arr, _ = roc_curve(self.y_true, self.y_proba)
        roc_auc_val = auc(fpr_arr, tpr_arr)
        pr_auc_val = average_precision_score(self.y_true, self.y_proba)

        # FPR @ target TPR
        fpr_at_target_tpr, thresh_at_tpr = compute_fpr_at_fixed_tpr(
            self.y_true, self.y_proba, target_tpr=target_tpr
        )

        # Spécificité / TNR
        specificity = tn / negatives if negatives > 0 else 0.0

        return {
            "Model Name": self.model_name,
            "Threshold": self.threshold,
            "Accuracy": acc,
            "Precision": prec,
            "Recall (TPR)": rec,
            "Specificity (TNR)": specificity,
            "F1-Score": f1,
            "F2-Score (Recall-focused)": f2,
            "F0.5-Score (Precision-focused)": f05,
            "MCC": mcc,
            "ROC-AUC": roc_auc_val,
            "PR-AUC (Average Precision)": pr_auc_val,
            f"FPR @ {int(target_tpr*100)}% Recall": fpr_at_target_tpr,
            f"Threshold @ {int(target_tpr*100)}% Recall": thresh_at_tpr,
            "Confusion Matrix": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "Prevalence (Phishing %)": (positives / total) * 100
            }
        }

    def print_summary(self, target_tpr: float = 0.95) -> None:
        """Affiche un résumé soigné dans la console."""
        metrics = self.compute_all_metrics(target_tpr=target_tpr)
        cm = metrics["Confusion Matrix"]

        print("=" * 65)
        print(f"📊 RAPPORT D'ÉVALUATION CYBER-ML : {self.model_name}")
        print("=" * 65)
        print(f"• Seuil appliqué           : {self.threshold:.2f}")
        print(f"• Prévalence Phishing réel : {cm['Prevalence (Phishing %)']:.2f}%")
        print("-" * 65)
        print("🎯 MÉTRIQUES STANDARDS & BASELINE :")
        print(f"  - Accuracy               : {metrics['Accuracy']:.4f}")
        print(f"  - Précision              : {metrics['Precision']:.4f} (éviter fausses alertes)")
        print(f"  - Rappel (Recall / TPR)  : {metrics['Recall (TPR)']:.4f} (capturer les attaques)")
        print(f"  - F1-Score               : {metrics['F1-Score']:.4f} (moyenne harmonique P/R)")
        print("-" * 65)
        print("🛡️ MÉTRIQUES CYBERSÉCURITÉ AVANCÉES :")
        print(f"  - F2-Score (priorité FN) : {metrics['F2-Score (Recall-focused)']:.4f}")
        print(f"  - F0.5-Score (priorité FP: {metrics['F0.5-Score (Precision-focused)']:.4f}")
        print(f"  - MCC (robuste déséquil.): {metrics['MCC']:.4f} (entre -1 et +1)")
        print(f"  - PR-AUC (Average Prec.) : {metrics['PR-AUC (Average Precision)']:.4f} ⭐ (clé en prod)")
        print(f"  - ROC-AUC                : {metrics['ROC-AUC']:.4f} (optimiste si déséquilibré)")
        print(f"  - FPR @ {int(target_tpr*100)}% Recall        : {metrics[f'FPR @ {int(target_tpr*100)}% Recall']:.4f} (Seuil requis = {metrics[f'Threshold @ {int(target_tpr*100)}% Recall']:.3f})")
        print("-" * 65)
        print("🔍 MATRICE DE CONFUSION :")
        print(f"  [TN (Légitimes OK)]: {cm['TN']:<7} | [FP (Fausses Alertes)]: {cm['FP']:<7}")
        print(f"  [FN (Attaques Ratées)]: {cm['FN']:<5} | [TP (Phishing Bloqué)]: {cm['TP']:<7}")
        print("=" * 65)

    def plot_confusion_matrix(
        self,
        ax: Optional[plt.Axes] = None,
        cmap: str = "Blues"
    ) -> plt.Axes:
        """
        Affiche une matrice de confusion enrichie avec libellés métiers (TN, FP, FN, TP)
        et pourcentages par classe.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        cm = np.array([[tn, fp], [fn, tp]])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        labels = [
            [f"TN (Legitime sain)\n{tn:,}\n({cm_normalized[0,0]*100:.1f}%)",
             f"FP (Fausse alerte)\n{fp:,}\n({cm_normalized[0,1]*100:.1f}%)"],
            [f"FN (Phishing rate [DANGER])\n{fn:,}\n({cm_normalized[1,0]*100:.1f}%)",
             f"TP (Phishing bloque [OK])\n{tp:,}\n({cm_normalized[1,1]*100:.1f}%)"]
        ]

        sns.heatmap(
            cm,
            annot=np.array(labels),
            fmt="",
            cmap=cmap,
            cbar=False,
            ax=ax,
            xticklabels=["Predit Legitime (0)", "Predit Phishing (1)"],
            yticklabels=["Reel Legitime (0)", "Reel Phishing (1)"],
            annot_kws={"fontsize": 10, "fontweight": "normal"}
        )
        ax.set_title(f"Matrice de Confusion\n(Seuil = {self.threshold:.2f})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Classe Réelle", fontsize=11)
        ax.set_xlabel("Classe Prédite", fontsize=11)
        return ax

    def plot_roc_curve(
        self,
        ax: Optional[plt.Axes] = None,
        target_tpr: float = 0.95
    ) -> plt.Axes:
        """
        Trace la courbe ROC avec mise en évidence du point de fonctionnement (ex: 95% Recall).
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        roc_auc_val = auc(fpr, tpr)
        fpr_fixed, _ = compute_fpr_at_fixed_tpr(self.y_true, self.y_proba, target_tpr)

        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"{self.model_name} (AUC = {roc_auc_val:.4f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Aléatoire (AUC = 0.50)")

        # Point de fonctionnement cible
        ax.axhline(target_tpr, color="crimson", linestyle=":", alpha=0.7)
        ax.axvline(fpr_fixed, color="crimson", linestyle=":", alpha=0.7)
        ax.scatter([fpr_fixed], [target_tpr], color="crimson", zorder=5,
                   label=f"FPR = {fpr_fixed:.3f} @ {int(target_tpr*100)}% TPR")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
        ax.set_title("Courbe ROC (Receiver Operating Characteristic)", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        return ax

    def plot_pr_curve(
        self,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Trace la courbe Precision-Recall avec PR-AUC et baseline de prévalence.
        Métrique reine en détection d'intrusion/phishing déséquilibré.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba)
        pr_auc_val = average_precision_score(self.y_true, self.y_proba)
        baseline_prevalence = np.mean(self.y_true)

        ax.plot(recall, precision, color="#2ca02c", lw=2,
                label=f"{self.model_name} (PR-AUC / AP = {pr_auc_val:.4f})")
        ax.axhline(baseline_prevalence, color="red", linestyle="--",
                   label=f"Baseline Aléatoire ({baseline_prevalence*100:.1f}%)")

        # Marquer le point actuel du seuil
        curr_prec = precision_score(self.y_true, self.y_pred, zero_division=0)
        curr_rec = recall_score(self.y_true, self.y_pred, zero_division=0)
        ax.scatter([curr_rec], [curr_prec], color="purple", s=60, zorder=5,
                   label=f"Seuil actuel ({self.threshold:.2f}): P={curr_prec:.2f}, R={curr_rec:.2f}")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Rappel (Recall / TPR)", fontsize=11)
        ax.set_ylabel("Précision", fontsize=11)
        ax.set_title("Courbe Precision-Recall (PR)", fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.3)
        return ax

    def plot_threshold_analysis(
        self,
        ax: Optional[plt.Axes] = None,
        num_thresholds: int = 100
    ) -> plt.Axes:
        """
        Analyse les métriques clés (Précision, Rappel, F1, F2, MCC) en fonction
        du seuil de décision (0 à 1).
        Crucial pour calibrer le modèle selon le besoin métier (SOC / Anti-Spam).
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        thresholds = np.linspace(0.01, 0.99, num_thresholds)
        precs, recs, f1s, f2s, mccs = [], [], [], [], []

        for th in thresholds:
            pred = (self.y_proba >= th).astype(int)
            precs.append(precision_score(self.y_true, pred, zero_division=0))
            recs.append(recall_score(self.y_true, pred, zero_division=0))
            f1s.append(f1_score(self.y_true, pred, zero_division=0))
            f2s.append(fbeta_score(self.y_true, pred, beta=2.0, zero_division=0))
            mccs.append(matthews_corrcoef(self.y_true, pred))

        ax.plot(thresholds, precs, label="Précision", color="#1f77b4", lw=1.8)
        ax.plot(thresholds, recs, label="Rappel (Recall)", color="#ff7f0e", lw=1.8)
        ax.plot(thresholds, f1s, label="F1-Score", color="#2ca02c", lw=2, linestyle="-.")
        ax.plot(thresholds, f2s, label="F2-Score (pénalise FN)", color="#d62728", lw=2)
        ax.plot(thresholds, mccs, label="MCC", color="#9467bd", lw=1.8, linestyle=":")

        # Seuil optimal F2
        best_f2_idx = np.argmax(f2s)
        best_f2_th = thresholds[best_f2_idx]
        ax.axvline(best_f2_th, color="red", linestyle="--", alpha=0.6,
                   label=f"Optimum F2 (Seuil={best_f2_th:.2f})")

        # Marquer le seuil actuel
        ax.axvline(self.threshold, color="black", linestyle="-", alpha=0.5,
                   label=f"Seuil Actuel ({self.threshold:.2f})")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([-0.1, 1.05])
        ax.set_xlabel("Seuil de classification (Threshold)", fontsize=11)
        ax.set_ylabel("Score de la métrique", fontsize=11)
        ax.set_title("Optimisation du Seuil de Décision (Trade-off Cyber)", fontsize=12, fontweight="bold")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=9)
        ax.grid(alpha=0.3)
        return ax

    def plot_full_dashboard(
        self,
        target_tpr: float = 0.95,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Génère une planche de bord complète (2x2) réunissant les 4 vues indispensables :
        1. Matrice de Confusion annotée
        2. Courbe ROC + ROC-AUC + FPR@TPR
        3. Courbe PR + PR-AUC + Seuil
        4. Sensibilité du seuil de décision (P, R, F1, F2, MCC)
        """
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Matrice de confusion
        self.plot_confusion_matrix(ax=axes[0, 0])

        # 2. ROC Curve
        self.plot_roc_curve(ax=axes[0, 1], target_tpr=target_tpr)

        # 3. PR Curve
        self.plot_pr_curve(ax=axes[1, 0])

        # 4. Threshold tuning
        self.plot_threshold_analysis(ax=axes[1, 1])

        fig.suptitle(
            f"Tableau de Bord d'Évaluation Modèle Phishing : {self.model_name}",
            fontsize=16,
            fontweight="bold",
            y=0.99
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Dashboard sauvegardé avec succès dans : {save_path}")

        return fig


def plot_comparison_chart(
    evaluation_results: List[Dict[str, Any]],
    metrics_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Trace un graphique comparatif en barres groupées pour comparer :
    - Reproduction Baseline (ex. Al-Subaiey et al., 2024)
    - Vos modèles (ex. TF-IDF + XGBoost vs RoBERTa)
    - Évaluation Cross-Corpus (ex. Train sur Enron/Nazario -> Test sur CEAS_08)
    - Robustesse sous Attaque Adversariale (ex. homoglyphes, injection de texte légitime)

    Args:
        evaluation_results: Liste de dictionnaires retournés par `compute_all_metrics()`.
        metrics_to_plot: Noms des métriques à comparer sur l'axe X.
        save_path: Chemin optionnel d'enregistrement de l'image.
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            "Precision",
            "Recall (TPR)",
            "F1-Score",
            "F2-Score (Recall-focused)",
            "MCC",
            "PR-AUC (Average Precision)"
        ]

    rows = []
    for res in evaluation_results:
        m_name = res["Model Name"]
        for met in metrics_to_plot:
            if met in res:
                rows.append({
                    "Expérience": m_name,
                    "Métrique": met,
                    "Valeur": res[met]
                })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=df,
        x="Métrique",
        y="Valeur",
        hue="Expérience",
        palette="viridis",
        ax=ax
    )

    ax.set_ylim([0, 1.05])
    ax.set_title("Comparaison Multi-Modèles / Cross-Corpus / Robustesse Adversariale", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Modèle / Scénario", loc="lower right", fontsize=10)

    # Annoter les valeurs au-dessus des barres
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.2f}",
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom',
                        fontsize=8, color='black',
                        xytext=(0, 2), textcoords='offset points')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Comparatif sauvegardé avec succès dans : {save_path}")

    return fig
