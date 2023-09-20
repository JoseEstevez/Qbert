import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    file = "random-agent-results.csv"
    data = pd.read_csv(file, header=None)

    runs = data.to_numpy().flatten()
    runs = np.sort(runs)

    # Plotting a histogram

    plt.ion()

    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')

    n, bins, patches = ax.hist(runs, bins=np.arange(0, 30000, 250),
                               facecolor='C0', rwidth=.9)

    ax.set_xlabel('Score')
    ax.set_ylabel('# of Times Achieved')
    ax.set_title('Qbert Scores')
    ax.axis([1000, 30000, 0, 12])
    ax.grid(True)
    print(runs)
    plt.show()
