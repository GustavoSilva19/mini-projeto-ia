
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"


BREAST_CANCER_FILE = DATA_DIR / "wdbc.data"

DIABETES_FILE = DATA_DIR / "pima_diabetes.xlsx"


RANDOM_STATE = 42

N_SPLITS = 5


POPULATION_SIZE = 15

MAX_GENERATIONS = 30

MAX_IMMIGRATION = 1.0

MAX_EMIGRATION = 1.0

MUTATION_PROBABILITY = 0.005


EXPONENTIAL_LAMBDA = 0.2


WEIGHT_MIN = -1.0

WEIGHT_MAX = 1.0


def get_hidden_layer_size(number_of_features: int) -> int:
    

    return (2 * number_of_features) + 1

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR.mkdir(parents=True, exist_ok=True)
