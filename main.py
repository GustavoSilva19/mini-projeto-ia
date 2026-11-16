from src.experiments import run_all_experiments


def main():

    print("=" * 60)

    print("MINI-PROJETO DE INTELIGÊNCIA ARTIFICIAL")

    print("MLP otimizada por BBO e ExpBBO")

    print("=" * 60)

    print("\nIniciando experimentos...")

    results, summary = run_all_experiments()

    print("\nExperimentos finalizados!")

    print("\nArquivos gerados:")

    print("- results/metrics/all_results.csv")

    print("- results/metrics/summary_results.csv")


if __name__ == "__main__":
    main()
