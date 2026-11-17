from pathlib import Path
from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR


METRICS = {
    "accuracy": "Acurácia",
    "precision": "Precisão",
    "sensitivity": "Sensibilidade",
    "specificity": "Especificidade",
    "f1_score": "F1-score",
    "g_mean": "G-Mean",
    "roc_auc": "ROC-AUC",
}

DATASET_NAMES = {
    "breast_cancer": "Câncer de Mama",
    "diabetes": "Diabetes",
}

CONFIGURATIONS = [
    ("standard", False, "BBO"),
    ("standard", True, "BBO + SMOTE"),
    ("exponential", False, "Exp-BBO"),
    ("exponential", True, "Exp-BBO + SMOTE"),
]

def format_number_br(value, decimals=3):
    return (
        f"{value:,.{decimals}f}"
        .replace(",", "TEMP")
        .replace(".", ",")
        .replace("TEMP", ".")
    )

def load_results():
    all_path = METRICS_DIR / "all_results.csv"
    summary_path = METRICS_DIR / "summary_results.csv"

    if not all_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "Os arquivos CSV não foram encontrados. "
            "Verifique a pasta results/metrics."
        )

    all_results = pd.read_csv(all_path)
    summary = pd.read_csv(summary_path)

    for df in (all_results, summary):
        if "smote" in df.columns:
            df["smote"] = (
                df["smote"]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False})
            )

    return all_results, summary


def get_configuration_data(df, dataset):
    subset = df[df["dataset"] == dataset].copy()

    subset["configuration"] = subset.apply(
        lambda row: (
            f"{'Exp-BBO' if row['algorithm'] == 'exponential' else 'BBO'}"
            f"{' + SMOTE' if row['smote'] else ''}"
        ),
        axis=1,
    )

    order = [item[2] for item in CONFIGURATIONS]

    subset["configuration"] = pd.Categorical(
        subset["configuration"],
        categories=order,
        ordered=True,
    )

    return subset.sort_values("configuration")


def plot_metric_comparison(summary, metric):
    column = f"{metric}_mean"

    if column not in summary.columns:
        print(f"Métrica indisponível: {column}")
        return

    datasets = ["breast_cancer", "diabetes"]
    x = np.arange(len(CONFIGURATIONS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))

    for index, dataset in enumerate(datasets):
        subset = get_configuration_data(summary, dataset)

        values = [
            subset.loc[
                subset["configuration"] == label, column
            ].iloc[0]
            for _, _, label in CONFIGURATIONS
        ]

        positions = x + (index - 0.5) * width

        bars = ax.bar(
            positions,
            values,
            width,
            label=DATASET_NAMES[dataset],
        )

        ax.bar_label(
            bars,
            labels=[
                format_number_br(bar.get_height())
                for bar in bars
            ],
            padding=3,
            fontsize=8,
        )

    ax.set_title(f"Comparação de {METRICS[metric]}")
    ax.set_ylabel(METRICS[metric])
    ax.set_xlabel("Configuração")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [item[2] for item in CONFIGURATIONS],
        rotation=10,
    )
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: format_number_br(value, 2))
    )

    fig.tight_layout()

    output = FIGURES_DIR / f"comparison_{metric}.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"Gráfico salvo: {output}")


def plot_smote_impact(summary):
    metrics = ["accuracy", "f1_score", "roc_auc"]

    for dataset in DATASET_NAMES:
        subset = summary[summary["dataset"] == dataset]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(metrics))
        width = 0.2

        for index, (algorithm, smote, label) in enumerate(
            CONFIGURATIONS
        ):
            row = subset[
                (subset["algorithm"] == algorithm)
                & (subset["smote"] == smote)
            ]

            if row.empty:
                raise ValueError(
                    f"Configuração ausente: {dataset}, {label}"
                )

            values = [
                row[f"{metric}_mean"].iloc[0]
                for metric in metrics
            ]

            positions = x + (index - 1.5) * width

            ax.bar(
                positions,
                values,
                width,
                label=label,
            )

        ax.set_title(
            f"Impacto do SMOTE - {DATASET_NAMES[dataset]}"
        )
        ax.set_ylabel("Valor médio")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METRICS[metric] for metric in metrics]
        )
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: format_number_br(value, 2))
        )

        fig.tight_layout()

        output = FIGURES_DIR / f"smote_impact_{dataset}.png"
        fig.savefig(output, dpi=200)
        plt.close(fig)

        print(f"Gráfico salvo: {output}")


def plot_confusion_matrices(all_results):
    required = {
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    }

    if not required.issubset(all_results.columns):
        print(
            "Matrizes de confusão indisponíveis: "
            "contagens não encontradas no CSV."
        )
        return

    for dataset in DATASET_NAMES:
        subset = all_results[
            all_results["dataset"] == dataset
        ]

        fig, axes = plt.subplots(
            2, 2, figsize=(10, 9)
        )

        for ax, (algorithm, smote, label) in zip(
            axes.flat, CONFIGURATIONS
        ):
            rows = subset[
                (subset["algorithm"] == algorithm)
                & (subset["smote"] == smote)
            ]

            counts = rows[
                [
                    "true_negative",
                    "false_positive",
                    "false_negative",
                    "true_positive",
                ]
            ].sum()

            matrix = np.array([
                [counts["true_negative"], counts["false_positive"]],
                [counts["false_negative"], counts["true_positive"]],
            ], dtype=int)

            ax.imshow(matrix, cmap="Blues")

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j, i, str(matrix[i, j]),
                        ha="center",
                        va="center",
                        fontsize=13,
                    )

            ax.set_title(label)
            ax.set_xlabel("Classe prevista")
            ax.set_ylabel("Classe real")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

        fig.suptitle(
            f"Matrizes de Confusão - "
            f"{DATASET_NAMES[dataset]}"
        )
        fig.tight_layout()

        output = (
            FIGURES_DIR / f"confusion_matrix_{dataset}.png"
        )
        fig.savefig(output, dpi=200)
        plt.close(fig)

        print(f"Gráfico salvo: {output}")


def generate_all_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results, summary = load_results()

    print("Gerando gráficos comparativos...")

    for metric in METRICS:
        plot_metric_comparison(summary, metric)

    print("Gerando gráficos de impacto do SMOTE...")
    plot_smote_impact(summary)

    print("Gerando matrizes de confusão...")
    plot_confusion_matrices(all_results)

    print("\nGráficos gerados com sucesso!")


if __name__ == "__main__":
    generate_all_figures()
