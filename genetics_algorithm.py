import numpy as np

from utils import get_static_wallet_value


class GeneticAlgorithm:
    def __init__(self, prices, assets_names, initial_cash, monthly_contribution):
        self.assets_names = assets_names
        self.prices = prices
        self.initial_cash = initial_cash
        self.monthly_contribution = monthly_contribution
        self.contribution_periods = len(prices)

        self.gen_max_value = 100

    def init_array_randomly(self, n):
        return [self.init_gen(self.gen_max_value) for _ in range(n)]

    @staticmethod
    def init_gen(gen_max_value=100):
        return np.random.randint(0, gen_max_value+1)

    def init_population(self, population_size, genes_count):
        return [self.init_array_randomly(genes_count)for _ in range(population_size)]

    @staticmethod
    def get_assets_amount(gen_normalized, prices_sample, cash):
        assets_values = np.multiply(gen_normalized, cash)
        assets_amounts = np.divide(assets_values, prices_sample)
        return assets_amounts

    @staticmethod
    def normalize_gen(arr):
        """distributes the values of the array so that their sum equals to 1"""
        arr_sum = np.sum(arr)
        normalized_arr = arr / arr_sum
        return normalized_arr

    def fitness(self, gen):
        """
        [0] - shares -> nasdaq
        [1] - gold -> SPDR Gold Trust
        [2] - currencies -> $
        [3] - bonds -> government bonds
        [4] - cash -> in the home safe
        """

        static_wallet_value = self.initial_cash
        gen_normalized = self.normalize_gen(gen)
        assets_amounts = self.get_assets_amount(gen_normalized, self.prices.iloc[0], static_wallet_value)

        # simulate monthly contribution and repurchase
        for _, row in self.prices.iterrows():
            contribution_amounts = self.get_assets_amount(gen_normalized, row, self.monthly_contribution)
            assets_amounts = [a + b for a, b in zip(assets_amounts, contribution_amounts)]
            static_wallet_value += self.monthly_contribution

        final_value = sum([i * j for i, j in zip(assets_amounts, self.prices.iloc[-1].tolist())])

        return final_value - static_wallet_value

    def selection(self, population):
        population_length = len(population)
        best = 0
        best_fitness = self.fitness(population[best])
        for i in range(1, population_length):
            fitness_i = self.fitness(population[i])
            if best_fitness < fitness_i:
                best = i
                best_fitness = fitness_i
        return best

    def crossover(self, population, best, pr_cro, pr_mut):
        population_length = len(population)
        genes_count = len(population[0])

        for i in range(population_length):
            if i != best:
                for j in range(genes_count):
                    rand_val = np.random.random()
                    if rand_val <= pr_cro:
                        population[i][j] = population[best][j]
                    elif rand_val <= pr_mut:
                        population[i][j] = np.random.randint(0, self.gen_max_value+1)
        return population

    def print_population_with_fitness(self, population):
        fitness_values = [self.fitness(gen) for gen in population]
        for i, obj_fitness in enumerate(fitness_values):
            print(f"Fitness of wallet {i + 1}: {obj_fitness:.2f}")

    def print_best(self, best_of_all):
        best_of_all_normalized = self.normalize_gen(best_of_all)
        print('Best wallet contains:')
        for gen, asset in zip(best_of_all_normalized, self.assets_names):
            print(f'{asset}: {gen:.2f} %')

        gain = self.fitness(best_of_all)
        static_wallet_value = get_static_wallet_value(self.initial_cash, self.monthly_contribution, self.contribution_periods)
        print(f'Wallet gains {gain:.2f}, that is {(gain/static_wallet_value*100):.2f} % over time')
