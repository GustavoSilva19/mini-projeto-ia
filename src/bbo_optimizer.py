import numpy as np

from src.config import (
    POPULATION_SIZE,
    MAX_GENERATIONS,
    MAX_IMMIGRATION,
    MAX_EMIGRATION,
    MUTATION_PROBABILITY,
    EXPONENTIAL_LAMBDA,
    WEIGHT_MIN,
    WEIGHT_MAX,
    RANDOM_STATE
)


class BBOOptimizer:

    def __init__(
        self,
        objective_function,
        dimension,
        variant="standard",
        population_size=POPULATION_SIZE,
        generations=MAX_GENERATIONS,
        lower_bound=WEIGHT_MIN,
        upper_bound=WEIGHT_MAX,
        random_state=RANDOM_STATE
    ):

        if variant not in ("standard", "exponential"):
            raise ValueError(
                "Variante deve ser 'standard' ou 'exponential'."
            )

        self.objective_function = objective_function
        self.dimension = dimension
        self.variant = variant

        self.population_size = population_size
        self.generations = generations

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.rng = np.random.default_rng(random_state)

        self.best_solution = None
        self.best_fitness = float("inf")

        self.history = []

    def initialize_population(self):

        return self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(self.population_size, self.dimension)
        )

    def evaluate_population(self, population):

        return np.array([
            self.objective_function(individual)
            for individual in population
        ])

    def calculate_migration_rates(self, fitness):

        ranking = np.argsort(fitness)

        species = np.empty(self.population_size, dtype=int)

        species[ranking] = np.arange(self.population_size)

        max_species = max(self.population_size - 1, 1)

        immigration = MAX_IMMIGRATION * (
            species / max_species
        )

        emigration = MAX_EMIGRATION * (
            1 - species / max_species
        )

        return immigration, emigration, species

    def migration_random_value(self):


        if self.variant == "standard":
            return self.rng.uniform(0.0, 1.0)


        u = self.rng.uniform(0.0, 1.0)

        lam = EXPONENTIAL_LAMBDA

        return -np.log(
            1 - u * (1 - np.exp(-lam))
        ) / lam

    def select_donor(self, emigration):


        total = np.sum(emigration)

        if total <= 0:
            probabilities = np.ones(
                self.population_size
            ) / self.population_size
        else:
            probabilities = emigration / total

        return self.rng.choice(
            self.population_size,
            p=probabilities
        )

    def migrate(
        self,
        population,
        immigration,
        emigration
    ):


        new_population = population.copy()

        for i in range(self.population_size):

            for gene in range(self.dimension):

                random_value = self.migration_random_value()

                if random_value < immigration[i]:

                    donor = self.select_donor(emigration)

                    if donor != i:

                        new_population[i, gene] = (
                            population[donor, gene]
                        )

        return new_population

    def mutate(self, population, species):


        new_population = population.copy()

        max_species = max(self.population_size - 1, 1)

        for i in range(self.population_size):

            mutation_rate = MUTATION_PROBABILITY * (
                1 - species[i] / max_species
            )

            for gene in range(self.dimension):

                if self.rng.random() < mutation_rate:

                    new_population[i, gene] = self.rng.uniform(
                        self.lower_bound,
                        self.upper_bound
                    )

        return new_population

    def optimize(self):


        population = self.initialize_population()

        fitness = self.evaluate_population(population)

        best_index = np.argmin(fitness)

        self.best_solution = population[best_index].copy()

        self.best_fitness = fitness[best_index]

        for generation in range(self.generations):

            immigration, emigration, species = (
                self.calculate_migration_rates(fitness)
            )


            new_population = self.migrate(
                population,
                immigration,
                emigration
            )


            new_population = self.mutate(
                new_population,
                species
            )


            new_fitness = self.evaluate_population(new_population)

            current_best_index = np.argmin(new_fitness)

            if new_fitness[current_best_index] < self.best_fitness:

                self.best_fitness = new_fitness[current_best_index]

                self.best_solution = (
                    new_population[current_best_index].copy()
                )

            worst_index = np.argmax(new_fitness)

            new_population[worst_index] = self.best_solution.copy()

            new_fitness[worst_index] = self.best_fitness

            population = new_population

            fitness = new_fitness

            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness