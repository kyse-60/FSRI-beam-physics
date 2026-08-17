import matplotlib.pyplot as plt
import pandas as pd
import ast

FILE = "outputValpha(100)-time-1.8745821999909822s.csv"
type = "alpha" #regular, alpha, 4eq
FIGNAME = "trial2.jpg"
SAMPLES = 4

#print(df.columns.tolist())

def makeplot(filename, figname,samples):
    df = pd.read_csv(filename)
    coord_cols = [f"coord{i}" for i in range(11)]
    sample = df.sample(n=samples)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    for index, row in sample.iterrows():
        points = [ast.literal_eval(row[c]) for c in coord_cols]
        xs, ys = zip(*points)
        if type == "regular":
            ax.plot(xs, ys, marker="o", label=f"F={row['F']:.3g}, E={row['E']:.3g}, base={row['base']:.3g}, height={row['height']:.3g}, L={row['length']:.3g}")
        if type == "alpha":
            ax.plot(xs, ys, marker="o", label=f"alpha={row['alpha']:.3g}")
        if type == "4eq":
            return 
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("4 random rows")
    plt.savefig(figname)

makeplot(FILE, FIGNAME, SAMPLES)
