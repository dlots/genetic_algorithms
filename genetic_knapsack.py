import random


def knapsack_ga(weights, prices, W, population_size, mut_freq, max_gener_numb, percent):
    generation = 0
    population = generate_population(weights, population_size)
    fitness = get_fitness(population, weights, prices, W)

    while (generation < max_gener_numb and not test(fitness, percent)):
        population = new_generation(population, fitness, mut_freq)
        generation += 1
        fitness = get_fitness(population, weights, prices, W)

    ans = select_best(population, fitness)
    return ans


def generate_population(weights, population_size):  # checked
    population = [[random.randint(0, 1) for i in range(len(weights))] for j in range(population_size)]
    return population


def get_fitness(population, weights, prices, W):  # checked
    fitness = []
    for i in range(len(population)):
        tmp = W + 1
        while (tmp > W):
            weight = 0
            tmp = 0
            mas = []
            for j in range(len(population[i])):
                if population[i][j] == 1:
                    mas += [j]
                    weight += prices[j]
                    tmp += weights[j]
            if tmp > W:
                ind = mas[random.randint(0, len(mas) - 1)]
                population[i][ind] = 0
        fitness += [weight]
    return fitness


def new_generation(population, fitness, mut_freq):  # checkedc
    generation = []
    generation += [select_best(population, fitness)]
    # print generation
    while (len(generation) < len(population)):
        chro1, chro2 = select(population, fitness)
        new_chro = crossover(chro1, chro2)
        generation += [mutation(new_chro, mut_freq)]
    return generation


def mutation(chro, mut_freq):  # checked
    for i in range(len(chro)):
        rnd = random.randint(1, mut_freq)
        if rnd == 1:  # mutate
            if (chro[i] == 1):
                chro[i] = 0
            else:
                chro[i] = 1
    return chro


def select(population, fitness):  # checked
    rnd = random.randint(0, sum(fitness))
    tmp = 0
    chro1 = []
    fit = 0
    for i in range(len(population)):
        tmp += fitness[i]
        if rnd <= tmp:
            fit = fitness.pop(i)
            chro1 = population.pop(i)
            break
    tmp = 0
    rnd = random.randint(0, sum(fitness))
    for i in range(len(population)):
        tmp += fitness[i]
    if rnd <= tmp:
        chro2 = population[i]
        population += [chro1]
        fitness += [fit]
        return (chro1, chro2)


def select_best(population, fitness):  # checkde
    index = 0
    for i in range(len(fitness)):
        if fitness[i] > fitness[index]:
            index = i
    return population[index]


def crossover(chro1, chro2):  # checked
    cross_point = random.randint(0, len(chro1) - 1)
    new_chro = chro1[:cross_point] + chro2[cross_point:]
    return new_chro


def get_max_cnt(fitness):  # checked
    max_cnt = 0
    unique_values = set(fitness)
    for i in unique_values:
        i_in_fit = fitness.count(i)
        if i_in_fit > max_cnt:
            max_cnt = i_in_fit
    return max_cnt


def test(fitness, percent):  # checked
    max_cnt = get_max_cnt(fitness)
    if max_cnt / len(fitness) < percent:
        return False
    else:
        return True
