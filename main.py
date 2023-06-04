from tqdm import tqdm

from data_manager import create_data
from genetics_algorithm import GeneticAlgorithm
from utils import get_static_wallet_value, months_between_dates

# data
tickers = ['^IXIC', 'GLD', 'USDPLN=X']
start_date = '2020-06-03'
end_date = '2023-06-03'
ASSETS_NAMES = ['NASDAQ', 'Gold', 'USD/PLN', 'Bonds', 'Cash']

# algorithm config
POPULATION_SIZE = 100
GENES_COUNT = 5
CROSSOVER_PROBABILITY = 0.65
MUTATION_PROBABILITY = 0.2
GENERATION_COUNT = 200

# finance config
INITIAL_CASH = 1000
MONTHLY_CONTRIBUTION = 100
CONTRIBUTION_PERIODS = months_between_dates(start_date, end_date)
EXPECTED_REWARD_PERCENTAGE = 20

if __name__ == '__main__':

    expected_reward_in_cash = \
        get_static_wallet_value(INITIAL_CASH, MONTHLY_CONTRIBUTION, CONTRIBUTION_PERIODS)*EXPECTED_REWARD_PERCENTAGE/100
    data = create_data(tickers, start_date, end_date, ASSETS_NAMES)

    ga = GeneticAlgorithm(data, ASSETS_NAMES, INITIAL_CASH, MONTHLY_CONTRIBUTION)
    population = ga.init_population(POPULATION_SIZE, GENES_COUNT)

    # ga.print_population_with_fitness(population)

    highest_gain = 0
    best_index = 0
    found_expected_best = False
    with tqdm(total=GENERATION_COUNT) as pbar:
        for generation_index in range(GENERATION_COUNT):

            best_index = ga.selection(population)
            highest_gain = ga.fitness(population[best_index])

            pbar.set_postfix({'highest gain': highest_gain}, refresh=True)
            pbar.update(1)

            if highest_gain >= expected_reward_in_cash:
                found_expected_best = True
                break

            population = ga.crossover(population, best_index, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY)

    print('=== Founded expected best ===' if found_expected_best else '=== Didn\'t find expected best ===')
    ga.print_best(population[best_index])

# Best wallet contains:
# NASDAQ: 0.80 %
# Gold: 0.03 %
# USD/PLN: 0.00 %
# Bonds: 0.15 %
# Cash: 0.03 %
# Wallet gains 272.29, that is 12.38 % over time

# NASDAQ: 0.90 %
# Gold: 0.03 %
# USD/PLN: 0.07 %
# Bonds: 0.01 %
# Cash: 0.00 %
# Wallet gains 285.86, that is 12.99 % over time
