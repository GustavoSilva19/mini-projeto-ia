
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

from src.config import (
    N_SPLITS,
    RANDOM_STATE,
    get_hidden_layer_size,
    METRICS_DIR
)

from src.data_preprocessing import (
    load_all_datasets,
    preprocess_fold,
    describe_dataset
)

from src.mlp_model import MLP

from src.bbo_optimizer import BBOOptimizer

from src.evaluation import (
    calculate_metrics,
    summarize_results
)


def run_single_experiment(
    X_train,
    y_train,
    X_validation,
    y_validation,
    algorithm,
    use_smote,
    random_state
):

    start_time = time.perf_counter()

    
    X_train_processed, X_validation_processed = preprocess_fold(
        X_train,
        X_validation
    )

    y_train_processed = np.asarray(y_train, dtype=int)

    y_validation_processed = np.asarray(
        y_validation,
        dtype=int
    )

    
    if use_smote:

        smote = SMOTE(
            random_state=random_state,
            k_neighbors=min(
                5,
                int(pd.Series(y_train_processed).value_counts().min()) - 1
            )
        )

        X_train_processed, y_train_processed = smote.fit_resample(
            X_train_processed,
            y_train_processed
        )

    
    input_size = X_train_processed.shape[1]

    hidden_size = get_hidden_layer_size(input_size)

    model = MLP(
        input_size=input_size,
        hidden_size=hidden_size
    )

    
    
    def objective_function(parameters):

        return model.calculate_mse(
            X_train_processed,
            y_train_processed,
            parameters
        )

    
    optimizer = BBOOptimizer(
        objective_function=objective_function,
        dimension=model.parameter_count,
        variant=algorithm,
        random_state=random_state
    )

    best_parameters, best_mse = optimizer.optimize()

    
    probabilities = model.predict_proba(
        X_validation_processed,
        best_parameters
    )

    predictions = (probabilities >= 0.5).astype(int)

    metrics = calculate_metrics(
        y_validation_processed,
        predictions,
        probabilities
    )

    validation_mse = float(
        np.mean(
            (y_validation_processed - probabilities) ** 2
        )
    )

    execution_time = time.perf_counter() - start_time

    metrics["mse"] = validation_mse

    metrics["training_mse"] = best_mse

    metrics["execution_time"] = execution_time

    metrics["convergence_history"] = optimizer.history

    return metrics


def run_all_experiments():
    

    datasets = load_all_datasets()

    results = []

    algorithms = [
        "standard",
        "exponential"
    ]

    smote_options = [
        False,
        True
    ]

    for dataset_name, (X, y) in datasets.items():

        describe_dataset(dataset_name, X, y)

        cross_validation = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        for algorithm in algorithms:

            for use_smote in smote_options:

                print("\n" + "=" * 60)

                print(f"Base: {dataset_name}")

                print(f"Algoritmo: {algorithm}")

                print(f"SMOTE: {'Sim' if use_smote else 'Não'}")

                print("=" * 60)

                for fold, (train_index, validation_index) in enumerate(
                    cross_validation.split(X, y),
                    start=1
                ):

                    print(f"\nExecutando Fold {fold}/{N_SPLITS}...")

                    X_train = X.iloc[train_index]

                    X_validation = X.iloc[validation_index]

                    y_train = y.iloc[train_index]

                    y_validation = y.iloc[validation_index]

                    metrics = run_single_experiment(
                        X_train=X_train,
                        y_train=y_train,
                        X_validation=X_validation,
                        y_validation=y_validation,
                        algorithm=algorithm,
                        use_smote=use_smote,
                        random_state=RANDOM_STATE + fold
                    )

                    result = {
                        "dataset": dataset_name,
                        "algorithm": algorithm,
                        "smote": use_smote,
                        "fold": fold
                    }

                    
                    result.update({
                        key: value
                        for key, value in metrics.items()
                        if key != "convergence_history"
                    })

                    results.append(result)

                    print(
                        f"Acurácia: {metrics['accuracy']:.4f} | "
                        f"F1: {metrics['f1_score']:.4f} | "
                        f"AUC: {metrics['roc_auc']:.4f}"
                    )

    results_df = pd.DataFrame(results)

    summary_df = summarize_results(results_df)

    
    results_df.to_csv(
        METRICS_DIR / "all_results.csv",
        index=False
    )

    
    summary_df.to_csv(
        METRICS_DIR / "summary_results.csv",
        index=False
    )

    print("\n" + "=" * 60)

    print("RESUMO DOS EXPERIMENTOS")

    print("=" * 60)

    print(
        summary_df[
            [
                "dataset",
                "algorithm",
                "smote",
                "accuracy_mean",
                "f1_score_mean",
                "roc_auc_mean"
            ]
        ].to_string(index=False)
    )

    return results_df, summary_df
