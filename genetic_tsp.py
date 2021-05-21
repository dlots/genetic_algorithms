import numpy as np
from collections import defaultdict
import random
from math import sqrt

DEBUG = True
USE_CROSSOVER = 3


def crossover1(mas1, mas2, p1, p2):  # p1 и p2 - левый и правый элемент "середины", которой гены обмениваются
    res1 = [-1 for i in range(len(mas1))]  # результирующие
    res2 = [-1 for i in range(len(mas2))]

    links = defaultdict(list)

    for i in range(p1, p2):
        res1[i] = mas2[i]
        res2[i] = mas1[i]
        links[mas1[i]] = mas2[i]  # создаем ссылки
        links[mas2[i]] = mas1[i]

    for i in range(len(res1)):
        if res1[i] == -1:
            if mas1[i] not in res1:
                res1[i] = mas1[i]
            else:
                res1[i] = links[mas1[i]]

            if mas2[i] not in res2:
                res2[i] = mas2[i]
            else:
                res2[i] = links[mas2[i]]

    return res1, res2


def crossover2(mas1, mas2):
    res1 = [-1 for i in range(len(mas1))]  # результирующие
    res2 = [-1 for i in range(len(mas2))]

    i = 0
    j = 1
    while i < len(res1) - j:
        if mas1[i] != mas2[i]:  # если значения на левом конце не совпадают
            if random.choices([mas1[i], mas2[i]]) == mas1[i]:
                if mas1[i] not in res1 and mas2[i] not in res2:
                    res1[i] = mas1[i]
                    res2[i] = mas2[i]
                else:
                    res1[i] = mas2[i]
                    res2[i] = mas1[i]
            else:
                if mas2[i] not in res1 and mas1[i] not in res2:
                    res1[i] = mas2[i]
                    res2[i] = mas1[i]
                else:
                    res1[i] = mas1[i]
                    res2[i] = mas2[i]
        else:
            res1[i] = mas1[i]
            res2[i] = mas2[i]

        if mas1[len(res1) - j] != mas2[len(res1) - j]:  # если значения на правом конце не совпадают
            if random.choices([mas1[len(res1) - j], mas2[len(res1) - j]]) == mas1[len(res1) - j]:
                if mas1[len(res1) - j] not in res1 and mas2[len(res1) - j] not in res2:
                    res1[len(res1) - j] = mas1[len(res1) - j]
                    res2[len(res1) - j] = mas2[len(res1) - j]
                else:
                    res1[len(res1) - j] = mas2[len(res1) - j]
                    res2[len(res1) - j] = mas1[len(res1) - j]
            else:
                if mas2[len(res1) - j] not in res1 and mas1[len(res1) - j] not in res2:
                    res1[len(res1) - j] = mas2[len(res1) - j]
                    res2[len(res1) - j] = mas1[len(res1) - j]
                else:
                    res1[len(res1) - j] = mas1[len(res1) - j]
                    res2[len(res1) - j] = mas2[len(res1) - j]
        else:
            res1[len(res1) - j] = mas1[len(res1) - j]
            res2[len(res1) - j] = mas2[len(res1) - j]

        i += 1
        j += 1

    if i == len(res1) - j:  # случай, если нечетное количество
        if random.choices([mas1[i], mas2[i]]) == mas1[i]:
            if mas1[i] not in res1 and mas2[i] not in res2:
                res1[i] = mas1[i]
                res2[i] = mas2[i]
            else:
                res1[i] = mas2[i]
                res2[i] = mas1[i]
        else:
            if mas2[i] not in res1 and mas1[i] not in res2:
                res1[i] = mas2[i]
                res2[i] = mas1[i]
            else:
                res1[i] = mas1[i]
                res2[i] = mas2[i]

    return res1, res2


def crossover3(ind1, ind2):
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


def crossover(first, second, p1=None, p2=None):
    if USE_CROSSOVER == 1:
        if p1 is None or p2 is None:
            raise ValueError('p1 and p2 cant be None if crossover1 is used')
        return crossover1(first, second, p1, p2)
    elif USE_CROSSOVER == 2:
        return crossover2(first, second)
    elif USE_CROSSOVER == 3:
        return crossover3(first, second)
    else:
        raise ValueError('Wrong USE_CROSSOVER setting')


def mutation(mas):
    choice = [i for i in range(len(mas))]
    position = random.choices(choice)[0]
    choice.remove(position)
    num = random.choices(choice)[0]
    mas[position], mas[num] = mas[num], mas[position]
    return mas


class GeneticTSP:
    def __init__(self, path, pool_size, selection_size, min_generations):
        self.pool_size = None
        self.selection_size_parameter = None
        self.points_coordinates = None
        self.points_distances = None
        self.number_of_points = None
        self.pool = None
        self.min_generations = None
        self.init(path, pool_size, selection_size, min_generations)
        self.generation = None
        self.fitness_list = None

    def init(self, path, pool_size, selection_size, min_generations):
        self.pool_size = pool_size
        if selection_size % 2 == 1:
            selection_size -= 1
        if selection_size < 4:
            selection_size = 4
        self.selection_size_parameter = selection_size
        self.min_generations = min_generations
        self.points_coordinates = []
        self.points_distances = {}
        self.number_of_points = 0
        file = open(path)
        read = False
        for line in file:
            if not read:
                if line == 'NODE_COORD_SECTION\n':
                    read = True
                continue
            if line == 'EOF\n':
                break
            point = line.strip().replace('\n', '').replace('   ', ' ').replace('  ', ' ').split(' ')
            self.points_coordinates.append((float(point[1]), float(point[2])))
            self.number_of_points += 1
        file.close()

    def get_distance(self, a, b):
        if a > b:
            a, b = b, a
        if (a, b) in self.points_distances:
            return self.points_distances[(a, b)]
        else:
            a_coordinates = self.points_coordinates[a]
            b_coordinates = self.points_coordinates[b]
            delta_x = b_coordinates[0] - a_coordinates[0]
            delta_y = b_coordinates[1] - a_coordinates[1]
            distance = sqrt(delta_x * delta_x + delta_y * delta_y)
            self.points_distances[(a, b)] = distance
            return distance

    def fitness(self, chromosome):
        value = 0
        length = len(chromosome)
        for gene_index in range(length - 1):
            value += self.get_distance(chromosome[gene_index], chromosome[gene_index + 1])
        value += self.get_distance(chromosome[-1], chromosome[0])
        return int(value)

    def circle(self, parents, num):
        # генерация вероятностей--------------------------------------------
        f_mas = []

        s = 0
        for p in range(self.pool_size):
            temp = 1 / self.fitness_list[p]
            s += temp
            f_mas.append(temp)

        probabilities = []

        prob = 0
        for i in f_mas:
            prob += i / s
            probabilities.append(prob)

        for i in range(len(probabilities)):
            probabilities[i] = round(probabilities[i], 2) * 100

        #print(probabilities)
        # вращение колеса--------------------------------------------------

        output = []
        for i in range(num):
            rand_num = random.randint(0, 100)
            #print(rand_num)
            previous = 0
            for j in range(len(probabilities)):
                if previous <= rand_num < probabilities[j]:
                    output.append(j)
                    break
                else:
                    previous = probabilities[j]
        return output

    def how_many_are_equal(self):
        count = {}
        for chromosome_index in range(self.pool_size):
            fitness = self.fitness_list[chromosome_index]
            if fitness in count:
                count[fitness] += 1
            else:
                count[fitness] = 1
        max_count = max(count.values())
        #print(count.keys())
        return max_count / self.pool_size

    def compute(self):
        self.pool = []
        self.fitness_list = [-1 for _ in range(self.pool_size)]
        self.generation = 0
        if DEBUG:
            random.seed(42)
        else:
            random.seed()
        for i in range(self.pool_size):
            self.pool.append(list(np.random.permutation(self.number_of_points)))
            self.fitness_list[i] = self.fitness(self.pool[i])
        #equals = self.how_many_are_equal()
        #while equals < 0.9 or self.generation < self.min_generations:
        while self.generation < self.min_generations:
            self.generation += 1
            #print(self.generation)
            selection = self.circle(self.pool, self.selection_size_parameter)
            selection_size = len(selection)
            if len(selection) % 2 == 1:
                selection.pop()
                selection_size -= 1
            for chromosome_index_index in range(0, selection_size, 2):
                first_index = selection[chromosome_index_index]
                second_index = selection[chromosome_index_index + 1]
                first_chromosome = self.pool[first_index]
                second_chromosome = self.pool[second_index]
                first_chromosome, second_chromosome = crossover(first_chromosome, second_chromosome, 20, 40)
                mutation_chance = 1000
                if random.randint(1, mutation_chance) == 1:
                    first_chromosome = mutation(first_chromosome)
                    #print('mutate')
                if random.randint(1, mutation_chance) == 1:
                    second_chromosome = mutation(second_chromosome)
                    #print('mutate')
                self.pool[first_index], self.pool[second_index] = first_chromosome, second_chromosome
                self.fitness_list[first_index] = self.fitness(first_chromosome)
                self.fitness_list[second_index] = self.fitness(second_chromosome)
                #equals = self.how_many_are_equal()
        min_value = min(self.fitness_list)
        solutions = [self.pool[chromosome_index] for chromosome_index in range(self.pool_size)
                     if self.fitness_list[chromosome_index] == min_value]
        unique_solutions = [list(x) for x in set(tuple(x) for x in solutions)]
        return min_value, unique_solutions


def tsp(path, pool_size, selection_size, min_generations):
    obj = GeneticTSP(path, pool_size, selection_size, min_generations)
    return obj.compute()


if __name__ == '__main__':
    test = 'a280.tsp'
    tsp = GeneticTSP(test, 50, 6, 1000)
    value, unique_solutions = tsp.compute()
    print('done')
    #one = [1,2,3,4,0]
    #two = [3,4,1,0,2]
    #print(crossover(one, two))