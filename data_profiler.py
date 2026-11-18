"""
data_profiler.py
----------------
Module d'analyse exploratoire, de profilage de données et de détection de biais
pour les corpus de détection d'e-mails de phishing (ML & NLP).

Fournit :
1. Analyse statistique complète : déséquilibre de classes, longueurs, doublons (data leakage),
   indicateurs d'ingénierie sociale (mots d'urgence, URLs, majuscules, balises HTML).
2. Diagnostic des biais de corpus (sur-représentation lexicale, séparation artificielle, templates dupliqués).
3. Moteur d'aide à la décision : Recommandation motivée des métriques à retenir et des architectures ML (Linéaire, SVM, Ensembles, Transformers).
4. Tableaux de bord graphiques exhaustifs haute résolution.
"""

import re
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# Mots-clés fréquents d'ingénierie sociale en cybersécurité
URGENCY_KEYWORDS = [
    "urgent", "immediately", "verify", "account", "suspended",
    "password", "security", "bank", "alert", "click", "access",
    "confirm", "update", "expire", "login", "action required"
]


class DatasetProfiler:
    """
    Analyseur de données et de biais pour les datasets d'emails de phishing.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        dataset_name: str = "Email Phishing Dataset"
    ):
        """
        Args:
            df: DataFrame contenant au moins la colonne de texte et le label binaire (0=légitime, 1=phishing).
            text_col: Nom de la colonne contenant le texte (corps + sujet éventuel).
            label_col: Nom de la colonne contenant le label binaire.
            dataset_name: Nom descriptif du dataset.
        """
        self.dataset_name = dataset_name
        self.text_col = text_col
        self.label_col = label_col

        # Copie de travail nettoyée des NaN dans le texte
        self.df = df.copy()
        self.df[self.text_col] = self.df[self.text_col].fillna("").astype(str)
        self.df[self.label_col] = pd.to_numeric(self.df[self.label_col], errors="coerce").fillna(0).astype(int)

        self._extract_lexical_features()

    def _extract_lexical_features(self) -> None:
        """Extrait des caractéristiques textuelles et heuristiques cyber pour chaque email."""
        # Longueur en caractères et en mots
        self.df["char_count"] = self.df[self.text_col].apply(len)
        self.df["word_count"] = self.df[self.text_col].apply(lambda s: len(s.split()))

        # Ratio de majuscules (indicateur d'insistance / cri)
        self.df["caps_ratio"] = self.df[self.text_col].apply(
            lambda s: (sum(1 for c in s if c.isupper()) / max(len(s), 1))
        )

        # Nombre de points d'exclamation
        self.df["exclamation_count"] = self.df[self.text_col].apply(lambda s: s.count("!"))

        # Présence ou nombre d'URLs dans le texte
        url_regex = re.compile(r"https?://|www\.", re.IGNORECASE)
        self.df["url_count"] = self.df[self.text_col].apply(lambda s: len(url_regex.findall(s)))

        # Présence de balises HTML suspectes (typiques des templates automatisés)
        html_regex = re.compile(r"<[^>]+>", re.IGNORECASE)
        self.df["html_tags_count"] = self.df[self.text_col].apply(lambda s: len(html_regex.findall(s)))

        # Compte de mots-clés d'urgence
        def count_urgency_words(text: str) -> int:
            text_lower = text.lower()
            return sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)

        self.df["urgency_keyword_count"] = self.df[self.text_col].apply(count_urgency_words)

    def compute_dataset_stats(self) -> Dict[str, Any]:
        """Calcule un profil statistique complet et détecte les risques de biais."""
        total_samples = len(self.df)
        val_counts = self.df[self.label_col].value_counts().to_dict()
        legit_count = val_counts.get(0, 0)
        phish_count = val_counts.get(1, 0)

        phish_ratio = (phish_count / total_samples) if total_samples > 0 else 0.0
        imbalance_ratio = (legit_count / phish_count) if phish_count > 0 else float("inf")

        # Doublons exacts (Data Leakage majeur en NLP)
        exact_text_duplicates = self.df.duplicated(subset=[self.text_col]).sum()
        dup_percentage = (exact_text_duplicates / total_samples) * 100 if total_samples > 0 else 0.0

        # Textes vides ou quasi-vides (< 5 mots)
        empty_texts = (self.df["word_count"] < 5).sum()

        # Statistiques par classe
        legit_df = self.df[self.df[self.label_col] == 0]
        phish_df = self.df[self.df[self.label_col] == 1]

        def get_class_summary(sub_df: pd.DataFrame) -> Dict[str, float]:
            if len(sub_df) == 0:
                return {}
            return {
                "avg_words": float(sub_df["word_count"].mean()),
                "median_words": float(sub_df["word_count"].median()),
                "avg_urls": float(sub_df["url_count"].mean()),
                "pct_with_urls": float((sub_df["url_count"] > 0).mean() * 100),
                "avg_caps_ratio": float(sub_df["caps_ratio"].mean() * 100),
                "avg_urgency_words": float(sub_df["urgency_keyword_count"].mean()),
                "pct_with_html": float((sub_df["html_tags_count"] > 0).mean() * 100)
            }

        stats = {
            "dataset_name": self.dataset_name,
            "total_samples": total_samples,
            "legit_count": legit_count,
            "phish_count": phish_count,
            "phish_percentage": phish_ratio * 100,
            "imbalance_ratio": imbalance_ratio,
            "duplicates_count": int(exact_text_duplicates),
            "duplicates_pct": dup_percentage,
            "empty_or_short_texts": int(empty_texts),
            "legit_summary": get_class_summary(legit_df),
            "phish_summary": get_class_summary(phish_df)
        }
        return stats

    def generate_recommendations(self) -> Dict[str, Any]:
        """
        Moteur d'aide à la décision : Analyse les propriétés du dataset
        et recommande les meilleures métriques et architectures de modèles adaptées.
        """
        stats = self.compute_dataset_stats()
        phish_pct = stats["phish_percentage"]
        total = stats["total_samples"]
        dup_pct = stats["duplicates_pct"]

        # 1. Recommandations de Métriques
        metric_recs = []
        metric_warnings = []

        if phish_pct < 20.0 or phish_pct > 80.0:
            metric_recs.extend([
                "⭐ PR-AUC (Average Precision) : OBLIGATOIRE. Le dataset présente un déséquilibre marqué. La PR-AUC reflète fidèlement la capacité à extraire le phishing minoritaire sans noyer l'analyste sous de faux positifs.",
                "⭐ MCC (Matthews Correlation Coefficient) : HAUTEMENT RECOMMANDÉ. Métrique symétrique $[-1, +1]$ résistante aux fortes disparités de volume.",
                "⭐ F2-Score : CONSEILLÉ pour refléter l'asymétrie de coût cyber (un phishing non détecté = incident grave, vs un faux positif = simple désagrément).",
                "⭐ FPR @ 95% TPR : STANDARD USENIX/IEEE S&P pour quantifier le taux de fausses alertes pour une protection garantie."
            ])
            metric_warnings.append(
                "⚠️ ACCURACY BANIE : Avec une classe minoritaire, prédire toujours la classe dominante donne une fausse impression d'excellence."
            )
            metric_warnings.append(
                "⚠️ ROC-AUC TROMPEUSE : Sur données déséquilibrées, le grand nombre de légitimes (TN) maintient artificiellement le FPR très bas, surévaluant le modèle."
            )
        else:
            metric_recs.extend([
                "⭐ F1-Score : Valable ici car le dataset de test est relativement équilibré (~50/50).",
                "⭐ ROC-AUC & PR-AUC : Les deux sont informatives et comparables à la littérature (ex: Al-Subaiey et al., 2024).",
                "⭐ Matrice de Confusion : Indispensable pour contrôler la parité FP / FN."
            ])

        # Avertissement Data Leakage
        if dup_pct > 5.0:
            metric_warnings.append(
                f"🚨 RISQUE DE FUITE DE DONNÉES (DATA LEAKAGE) : {dup_pct:.1f}% de doublons exacts détectés. Si ces doublons se retrouvent à la fois dans le Train et le Test, vos scores (Accuracy, F1) seront artificiellement surévalués de 5 à 15 points !"
            )

        # 2. Recommandations d'Architectures de Modèles
        model_recs = {
            "Modèles Linéaires (Régression Logistique L1/L2, SGDClassifier)": {
                "Adéquation": "Excellente (Baseline Standard)",
                "Pourquoi": "Très rapide, parfaitement adapté aux matrices TF-IDF creuses (50k+ dimensions). Les coefficients des mots sont directement interprétables pour auditer les faux biais (ex: détecter si le modèle classe un mail en phishing uniquement à cause d'une date ou d'un nom de domaine spécifique).",
                "Quand l'utiliser": "Comme référence obligatoire (Baseline Al-Subaiey 2024), tests de transfert rapide, inférence ultra-rapide en temps réel."
            },
            "Support Vector Machines (LinearSVC)": {
                "Adéquation": "Idéale pour le texte pur TF-IDF",
                "Pourquoi": "L'hyperplan à marge maximale gère admirablement la haute dimension textuelle et résiste bien au surapprentissage quand le nombre de features dépasse le nombre d'échantillons.",
                "Quand l'utiliser": "Pour battre la régression logistique sur le TF-IDF tout en restant extrêmement rapide à l'inférence."
            },
            "Ensembles d'Arbres (Random Forest, XGBoost, LightGBM)": {
                "Adéquation": "Excellente sur features hybrides / métadonnées",
                "Pourquoi": "Moins efficaces sur du texte brut très sparse (50k colonnes TF-IDF), mais imbattables si vous combinez le texte avec des métadonnées (nb URLs, ratio majuscules, présence HTML, domaine d'envoi, longueur).",
                "Quand l'utiliser": "Si vous développez un pipeline d'ingénierie de features (handcrafted cyber features)."
            },
            "Transformers Pré-entraînés (DistilRoBERTa, DeBERTa-v3)": {
                "Adéquation": "État de l'art sémantique",
                "Pourquoi": "Capables de comprendre le contexte subtil, les tournures d'ingénierie sociale complexes et le spear-phishing sans mot-clé évident.",
                "Attention Cyber": "Coût computationnel élevé, temps d'inférence plus lourd dans un pipeline de messagerie d'entreprise, vulnérables à des attaques de manipulation de tokens (homoglyphes / zero-width spaces)."
            }
        }

        return {
            "metric_recommendations": metric_recs,
            "metric_warnings": metric_warnings,
            "model_recommendations": model_recs
        }

    def print_profiling_report(self) -> None:
        """Affiche un rapport textuel complet et structuré dans la console."""
        stats = self.compute_dataset_stats()
        recs = self.generate_recommendations()

        print("=" * 70)
        print(f"📊 PROFILAGE DU DATASET & ANALYSE DE BIAIS : {self.dataset_name}")
        print("=" * 70)
        print(f"• Total d'échantillons      : {stats['total_samples']:,}")
        print(f"• Mails Légitimes (0)       : {stats['legit_count']:,} ({(100 - stats['phish_percentage']):.1f}%)")
        print(f"• Mails Phishing (1)        : {stats['phish_count']:,} ({stats['phish_percentage']:.1f}%)")
        print(f"• Ratio de déséquilibre     : 1 phishing pour {stats['imbalance_ratio']:.1f} légitimes")
        print(f"• Doublons exacts de texte  : {stats['duplicates_count']:,} ({stats['duplicates_pct']:.1f}%)")
        print(f"• Textes vides / très courts: {stats['empty_or_short_texts']:,}")
        print("-" * 70)
        print("🔍 COMPARATIF DES CARACTÉRISTIQUES CYBER :")
        l_s = stats["legit_summary"]
        p_s = stats["phish_summary"]
        if l_s and p_s:
            print(f"  - Longueur moyenne (mots) : Légitime = {l_s['avg_words']:.1f} | Phishing = {p_s['avg_words']:.1f}")
            print(f"  - Présence d'URLs         : Légitime = {l_s['pct_with_urls']:.1f}% | Phishing = {p_s['pct_with_urls']:.1f}%")
            print(f"  - Ratio de Majuscules     : Légitime = {l_s['avg_caps_ratio']:.1f}% | Phishing = {p_s['avg_caps_ratio']:.1f}%")
            print(f"  - Mots-clés d'Urgence     : Légitime = {l_s['avg_urgency_words']:.2f} | Phishing = {p_s['avg_urgency_words']:.2f} par mail")
            print(f"  - Présence de code HTML   : Légitime = {l_s['pct_with_html']:.1f}% | Phishing = {p_s['pct_with_html']:.1f}%")
        print("-" * 70)
        print("🎯 RECOMMANDATION DES MÉTRIQUES :")
        for m in recs["metric_recommendations"]:
            print(f"  {m}")
        if recs["metric_warnings"]:
            print("\n  ⚠️ AVERTISSEMENTS BIAIS & MÉTRIQUES :")
            for w in recs["metric_warnings"]:
                print(f"  {w}")
        print("-" * 70)
        print("🤖 RECOMMANDATION DES FAMILLES DE MODÈLES :")
        for m_name, m_info in recs["model_recommendations"].items():
            print(f"\n  🔹 {m_name}")
            print(f"     • Statut : {m_info['Adéquation']}")
            print(f"     • Pourquoi : {m_info['Pourquoi']}")
            if "Quand l'utiliser" in m_info:
                usage = m_info["Quand l'utiliser"]
                print(f"     • Usage recommandé : {usage}")
            if "Attention Cyber" in m_info:
                print(f"     • Alerte Cybersécurité : {m_info['Attention Cyber']}")
        print("=" * 70)

    def plot_data_profile_dashboard(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Génère une planche graphique riche (6 sous-graphiques) pour auditer visuellement
        le dataset, ses distributions et ses biais.
        """
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        stats = self.compute_dataset_stats()

        # 1. Répartition des classes (Donut Chart)
        ax1 = axes[0, 0]
        labels = [f"Legitime ({stats['legit_count']:,})", f"Phishing ({stats['phish_count']:,})"]
        sizes = [stats["legit_count"], stats["phish_count"]]
        colors = ["#2b5c8f", "#d95f02"]
        wedges, texts, autotexts = ax1.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
        )
        for t in texts:
            t.set_fontsize(10)
        for at in autotexts:
            at.set_fontsize(10)
            at.set_weight("bold")
        ax1.set_title(f"1. Equilibre des Classes\n(Ratio: 1 / {stats['imbalance_ratio']:.1f})", fontsize=11, fontweight="bold")

        # 2. Distribution de la Longueur des Textes (Échelle Log)
        ax2 = axes[0, 1]
        df_plot = self.df[self.df["word_count"] > 0].copy()
        df_plot["label_str"] = df_plot[self.label_col].map({0: "Legitime", 1: "Phishing"})

        sns.histplot(
            data=df_plot,
            x="word_count",
            hue="label_str",
            bins=40,
            log_scale=True,
            palette={"Legitime": "#2b5c8f", "Phishing": "#d95f02"},
            kde=True,
            ax=ax2,
            element="step",
            common_norm=False
        )
        ax2.set_title("2. Longueur des Textes (Nb de mots, log-scale)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Nombre de mots (echelle log)", fontsize=10)
        ax2.set_ylabel("Densite / Frequence", fontsize=10)

        # 3. Indicateurs d'Ingénierie Sociale (Barres comparatives)
        ax3 = axes[0, 2]
        l_s = stats["legit_summary"]
        p_s = stats["phish_summary"]
        if l_s and p_s:
            features = ["Avec URLs (%)", "Avec HTML (%)", "Majuscules (%)", "Mots Urgence (x10)"]
            legit_vals = [l_s["pct_with_urls"], l_s["pct_with_html"], l_s["avg_caps_ratio"], l_s["avg_urgency_words"] * 10]
            phish_vals = [p_s["pct_with_urls"], p_s["pct_with_html"], p_s["avg_caps_ratio"], p_s["avg_urgency_words"] * 10]

            x_idx = np.arange(len(features))
            width = 0.35
            ax3.bar(x_idx - width/2, legit_vals, width, label="Legitime", color="#2b5c8f")
            ax3.bar(x_idx + width/2, phish_vals, width, label="Phishing", color="#d95f02")
            ax3.set_xticks(x_idx)
            ax3.set_xticklabels(features, fontsize=9, rotation=15)
            ax3.set_title("3. Heuristiques & Biais Lexicaux", fontsize=11, fontweight="bold")
            ax3.set_ylabel("Pourcentage / Taux", fontsize=10)
            ax3.legend(fontsize=9)

        # 4. Séparabilité Linéaire 2D (TruncatedSVD sur TF-IDF)
        ax4 = axes[1, 0]
        # Échantillonnage pour fluidité si grand dataset
        sample_size = min(1500, len(self.df))
        sample_df = self.df.sample(sample_size, random_state=42)
        try:
            tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
            tfidf_mat = tfidf.fit_transform(sample_df[self.text_col])
            svd = TruncatedSVD(n_components=2, random_state=42)
            coords = svd.fit_transform(tfidf_mat)

            scatter = ax4.scatter(
                coords[:, 0],
                coords[:, 1],
                c=sample_df[self.label_col].map({0: "#2b5c8f", 1: "#d95f02"}),
                alpha=0.45,
                s=18
            )
            ax4.set_title("4. Separabilite TF-IDF 2D (SVD / LSA)", fontsize=11, fontweight="bold")
            ax4.set_xlabel("Composante 1", fontsize=10)
            ax4.set_ylabel("Composante 2", fontsize=10)
            ax4.grid(alpha=0.3)
        except Exception as e:
            ax4.text(0.5, 0.5, f"Visualisation SVD non disponible:\n{e}", ha="center", va="center")

        # 5. Top Mots Fréquents & Signaux de Phishing
        ax5 = axes[1, 1]
        try:
            # Recherche des mots les plus corrélés à la classe phishing
            tfidf_kw = TfidfVectorizer(max_features=20, stop_words="english", ngram_range=(1, 1))
            X_phish = tfidf_kw.fit_transform(self.df[self.df[self.label_col] == 1][self.text_col])
            mean_tfidf = np.array(X_phish.mean(axis=0)).flatten()
            top_indices = np.argsort(mean_tfidf)[::-1][:10]
            feature_names = np.array(tfidf_kw.get_feature_names_out())[top_indices]
            scores = mean_tfidf[top_indices]

            sns.barplot(x=scores, y=feature_names, color="#d95f02", ax=ax5)
            ax5.set_title("5. Top 10 Mots Phishing (TF-IDF moyen)", fontsize=11, fontweight="bold")
            ax5.set_xlabel("Poids TF-IDF moyen", fontsize=10)
        except Exception as e:
            ax5.text(0.5, 0.5, f"Top mots non disponible:\n{e}", ha="center", va="center")

        # 6. Audit Qualité & Risques de Fuite (Data Leakage)
        ax6 = axes[1, 2]
        integrity_labels = ["Total Echantillons", "Textes Uniques", "Doublons Exacts", "Textes Courts (<5 mots)"]
        unique_texts = stats["total_samples"] - stats["duplicates_count"]
        integrity_vals = [stats["total_samples"], unique_texts, stats["duplicates_count"], stats["empty_or_short_texts"]]
        bars = ax6.barh(integrity_labels, integrity_vals, color=["#4daf4a", "#377eb8", "#e41a1c", "#ff7f00"])
        ax6.set_xscale("log")
        ax6.set_title("6. Integrite du Dataset & Fuite (Data Leakage)", fontsize=11, fontweight="bold")
        ax6.set_xlabel("Nombre d'e-mails (echelle log)", fontsize=10)
        for bar in bars:
            val = bar.get_width()
            ax6.text(val * 1.08, bar.get_y() + bar.get_height()/2, f"{int(val):,}",
                     va="center", fontsize=9, fontweight="bold")

        fig.suptitle(
            f"Audit Statistique, Profilage & Détection de Biais : {self.dataset_name}",
            fontsize=15,
            fontweight="bold",
            y=0.99
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Dashboard de profilage sauvegardé avec succès dans : {save_path}")

        return fig

    def plot_model_recommendation_radar(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Génère un graphique Radar (toile d'araignée) comparant les 4 familles de modèles
        selon les critères critiques en détection de phishing :
        - Vitesse d'Inférence (temps réel pour le proxy mail)
        - Interprétabilité (explicabilité pour l'analyste SOC)
        - Robustesse aux Attaques (évasion adversariale)
        - Compréhension Sémantique (spear-phishing subtil)
        - Légèreté & Déploiement
        """
        categories = [
            "Vitesse Inférence",
            "Interprétabilité",
            "Robustesse Attaques",
            "Puissance Sémantique",
            "Légèreté Déploiement"
        ]
        N = len(categories)

        # Scores [0 à 5] des 4 architectures
        model_scores = {
            "Régression Log. / SGD": [5.0, 5.0, 3.0, 2.5, 5.0],
            "Support Vector Machine (LinearSVC)": [4.5, 4.0, 3.5, 3.0, 4.8],
            "Ensembles d'Arbres (XGBoost/RF)": [4.0, 3.5, 4.0, 3.0, 4.0],
            "Transformers (RoBERTa / DeBERTa)": [1.5, 1.5, 2.5, 5.0, 1.5]
        }

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Boucler le tracé

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories, fontsize=10, fontweight="bold")
        ax.set_rlabel_position(0)
        plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=8)
        plt.ylim(0, 5.5)

        colors = ["#377eb8", "#4daf4a", "#984ea3", "#e41a1c"]
        for idx, (m_name, scores) in enumerate(model_scores.items()):
            scores_loop = scores + scores[:1]
            c = colors[idx % len(colors)]
            ax.plot(angles, scores_loop, linewidth=2, linestyle="solid", label=m_name, color=c)
            ax.fill(angles, scores_loop, color=c, alpha=0.12)

        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
        ax.set_title(
            "Benchmark Décisionnel : Quelle Famille de Modèle Choisir ?",
            fontsize=13,
            fontweight="bold",
            y=1.12
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Radar de décision sauvegardé avec succès dans : {save_path}")

        return fig
