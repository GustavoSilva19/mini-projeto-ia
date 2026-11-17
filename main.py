from src.experiments import run_all_experiments
from src.convergence import run_convergence_experiments
from src.visualization import generate_all_figures


def main():
    print("=" * 60)
    print("MINI-PROJETO DE INTELIGÊNCIA ARTIFICIAL")
    print("MLP otimizada por BBO e ExpBBO")
    print("=" * 60)

    print("\nIniciando experimentos principais...")
    results, summary = run_all_experiments()

    print("\nIniciando experimentos de convergência...")
    run_convergence_experiments()

    print("\nGerando gráficos dos resultados...")
    generate_all_figures()

    print("\nTodas as etapas foram finalizadas!")

    print("\nArquivos gerados:")
    print("- results/metrics/all_results.csv")
    print("- results/metrics/summary_results.csv")
    print("- results/metrics/convergence_history.csv")
    print("- 15 gráficos em results/figures/")


if __name__ == "__main__":
    main()
