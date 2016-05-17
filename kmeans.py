import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np

def generate_vector_set(num_points=2000):
    vectors_set = []
    for i in xrange(num_points):
        random_x = np.random.normal(3.0, 0.5)
        random_y = np.random.normal(1.0, 0.5)

        if np.random.random() > 0.5:
            random_x = np.random.normal(0.0, 0.9)
            random_y = np.random.normal(0.0, 0.9)

        vectors_set.append([random_x, random_y])
    return vectors_set


def generate_plot():
    vectors_set = generate_vector_set()
    df = pd.DataFrame({'x': [v[0] for v in vectors_set],
                       'y': [v[1] for v in vectors_set]})
    sns.lmplot('x', 'y', data=df, fit_reg=False, size=6)
    plt.show()

generate_plot()
