import numpy as np
import matplotlib.pyplot as plt

samples_per_cluster = 20
range_len = 10
var = 0.05

X = np.zeros(((range_len + 1) ** 2 * samples_per_cluster, 2))
y = np.zeros((range_len + 1) ** 2 * samples_per_cluster)
for x_cord in range(0, range_len + 1):
    for y_cord in range(0, range_len + 1):
        print(y_cord, x_cord)
        X[(y_cord + x_cord * (range_len + 1)) * samples_per_cluster:(y_cord + x_cord * (range_len + 1) + 1) * samples_per_cluster] = np.random.multivariate_normal(
            (x_cord - int(range_len / 2), y_cord - int(range_len / 2)), cov=[[var, 0], [0, var]], size=samples_per_cluster)
        y[(y_cord + x_cord * (range_len + 1)) * samples_per_cluster:(y_cord + x_cord * (range_len + 1) + 1) * samples_per_cluster] = (x_cord + y_cord) % 2

plt.plot(X[y == 0, 0], X[y == 0, 1], 'r.', label='0')
plt.plot(X[y == 1, 0], X[y == 1, 1], 'b.', label='1')
plt.legend()
plt.show()