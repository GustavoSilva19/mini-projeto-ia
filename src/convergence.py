
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split

from src.config import FIGURES_DIR, METRICS_DIR, RANDOM_STATE
from src.data_preprocessing import load_all_datasets
from src.experiments import run_single_experiment


DATASET_NAMES = {
    "breast_cancer": "Câncer de Mama",
    "diabetes": "Diabetes",
}

ALGORITHMS = {
    "standard": "BBO Padrão",
    "exponential": "BBO Exponencial",
}


def format_number_br(value, decimals=3):
    return (
        f"{value:,.{decimals}f}"
        .replace(",", "TEMP")
        .replace(".", ",")
        .replace("TEMP", ".")
    )


def run_convergence_experiments():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets()
    records = []

    for dataset_name, (X, y) in datasets.items():

        X_train, X_validation, y_train, y_validation = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=y,
                random_state=RANDOM_STATE,
            )
        )

        for use_smote in [False, True]:

            histories = {}

            for algorithm in ALGORITHMS:

                print(
                    f"Executando: {dataset_name} | "
                    f"{algorithm} | SMOTE={use_smote}",
                    flush=True,
                )

                result = run_single_experiment(
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    algorithm=algorithm,
                    use_smote=use_smote,
                    random_state=RANDOM_STATE,
                )

                history = result["convergence_history"]
                histories[algorithm] = history

                for generation, mse in enumerate(
                    history, start=1
                ):
                    records.append({
                        "dataset": dataset_name,
                        "algorithm": algorithm,
                        "smote": use_smote,
                        "generation": generation,
                        "training_mse": float(mse),
                    })

            plot_convergence(
                dataset_name,
                use_smote,
                histories,
            )

    output = METRICS_DIR / "convergence_history.csv"

    pd.DataFrame(records).to_csv(
        output,
        index=False,
    )

    print(f"\nHistóricos salvos em: {output}")
    print("Gráficos de convergência gerados!")


def plot_convergence(dataset_name, use_smote, histories):
    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm, history in histories.items():
        generations = np.arange(1, len(history) + 1)

        ax.plot(
            generations,
            history,
            linewidth=2,
            label=ALGORITHMS[algorithm],
        )

    smote_label = "Com SMOTE" if use_smote else "Sem SMOTE"

    ax.set_title(
        f"Convergência dos Algoritmos - "
        f"{DATASET_NAMES[dataset_name]}\n"
        f"{smote_label}"
    )

    ax.set_xlabel("Geração")
    ax.set_ylabel("Melhor MSE de treinamento")
    ax.grid(alpha=0.25)
    ax.legend()

    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: format_number_br(value, 3)
        )
    )

    fig.tight_layout()

    smote_suffix = "with_smote" if use_smote else "without_smote"

    output = (
        FIGURES_DIR
        / f"convergence_{dataset_name}_{smote_suffix}.png"
    )

    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"Gráfico salvo: {output}")


if __name__ == "__main__":
    run_convergence_experiments()
