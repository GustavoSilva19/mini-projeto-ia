
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