import numpy as np
from collections import defaultdict
import random
from math import sqrt


class GeneticTSP:
    def __init__(self, size, path):
        self.size = size
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
            self.points_coordinates.append((int(point[1]), int(point[2])))
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
        for gene_index in range(len(chromosome) - 1):
            value += self.get_distance(chromosome[gene_index], chromosome[gene_index + 1])
        return value

    def circle(self, parents, num):
        # генерация вероятностей--------------------------------------------
        f_mas = []

        s = 0
        for p in parents:
            temp = 1 / self.fitness(p)
            s += temp
            f_mas.append(temp)

        probabilities = []

        prob = 0
        for i in f_mas:
            prob += i / s
            probabilities.append(prob)

        for i in range(len(probabilities)):
            probabilities[i] = round(probabilities[i], 2) * 100

        print(probabilities)
        # вращение колеса--------------------------------------------------

        output = []
        for i in range(num):
            rand_num = random.randint(0, 100)
            print(rand_num)
            previous = 0
            for j in range(len(probabilities)):
                if previous <= rand_num < probabilities[j]:
                    output.append(j)
                    break
                else:
                    previous = probabilities[j]
        return output


if __name__ == '__main__':
    test = 'a280.tsp'
    tsp = GeneticTSP(1, test)
    print(tsp.fitness([1, 4, 5, 10, 6, 3, 167, 200]))
