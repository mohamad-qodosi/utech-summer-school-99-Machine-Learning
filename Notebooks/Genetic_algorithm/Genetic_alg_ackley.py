import numpy as np
import matplotlib.pyplot as plt

def ackley(X):
    p1 = -0.2 * np.sqrt(np.mean(X ** 2, axis=-1))
    p2 = np.mean(np.cos(2 * np.pi * X), axis=-1)

    t1 = -20 * np.exp(p1)
    t2 = np.exp(p2)

    return t1 - t2 + 20 + np.e


def parent_selection(fitness):
    sum_fitness = np.sum(fitness)
    selection_prob = fitness / sum_fitness
    idx = np.arange(fitness.shape[0])
    parent_idx = np.random.choice(idx, size=fitness.shape[0], p=selection_prob)
    return parent_idx

    # roulette role for one sample
    # t = np.random.random()
    # selection_prob_sorted = np.sort(selection_prob)
    # for i in range(selection_prob_sorted.shape[0]):
    #     if t < selection_prob_sorted[i]:
    #         print(i)
    #         break
    #     else:
    #         t = t - selection_prob_sorted[i]


def crossover(parents):
    offsprings = parents.copy()
    for i in range(0, parents.shape[0], 2):
        # crossover parents[i], parents[i + 1]
        mask = np.random.randint(0, 2, size=parents.shape[1])

        offsprings[i, mask == 1] = parents[i + 1, mask == 1]
        offsprings[i + 1, mask == 1] = parents[i, mask == 1]

    return offsprings


def mutation(offsprings):
    mask = np.random.random(size=offsprings.shape) > 0.7
    new_values = np.random.random(size=(1000, 10)) * 20 - 10

    offsprings[mask == 1] = new_values[mask == 1]
    return offsprings


population = np.random.random(size=(1000, 10)) * 20 - 10
best_fitness = []
mean_fitness = []
for i in range(1000):
    error = ackley(population)

    best_fitness.append(error.min())
    mean_fitness.append(error.mean())

    fitness = 20 - error

    parents_idx = parent_selection(fitness)
    parents = population[parents_idx]

    offsprings = crossover(parents)
    offsprings = mutation(offsprings)

    population = np.vstack((population, offsprings))
    population = np.array(sorted(population, key=ackley))[:1000]

print(population[0])
plt.plot(mean_fitness, 'b-')
plt.plot(best_fitness, 'r-')

plt.show()