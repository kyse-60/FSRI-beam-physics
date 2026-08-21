import matplotlib.pyplot as plt
import pandas as pd
import ast

equationFILE = "output4eq-time-37.722191700013354s.csv"
regularFILE = "outputV1.2-time-19.708904600003734s.csv"
FIGNAME = f'compare.1.jpg'
SAMPLES = 4
PARAMS = ['F', 'E', 'base', 'height', 'length']

def getpoints(row, num):
    coord_cols = [f"coord{i}" for i in range(num)]
    points = [ast.literal_eval(row[c]) for c in coord_cols]
    return zip(*points)

def makeplot(figname,samples):
    dfregular = pd.read_csv(regularFILE)
    dfequation = pd.read_csv(equationFILE)

    sample = dfregular.sample(n=samples)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (index, row) in zip(axes.flat, sample.iterrows()):
        xs, ys = getpoints(row, 11)
        ax.plot(xs, ys, marker="o", label="regular")
        eqrow = dfequation[(dfequation[PARAMS] == row[PARAMS]).all(axis=1)].iloc[0]

        xs, ys = getpoints(eqrow, 10)

        ax.plot(xs, ys, marker="s", label="equation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.set_title(f"F={row['F']:.3g}, E={row['E']:.3g}, b={row['base']:.3g}, h={row['height']:.3g}, L={row['length']:.3g}", fontsize=9)
    fig.tight_layout()
    plt.savefig(figname)
 

makeplot(FIGNAME, SAMPLES)
