import matplotlib.pyplot as plt
import pandas as pd
import ast

# FILE = "outputValpha(100)-time-1.8745821999909822s.csv"
# FILE = "output4eq-time-19.17769340000814s.csv"
FILE = "output4eq-time-28.227802200010046s.csv"
# FILE = "outputV1.1-time-18.980601800023578s.csv"
type = "4eq" #regular, alpha, 4eq
FIGNAME = "trial4eq.1.jpg"
SAMPLES = 4

def makeplot(filename, figname,samples):
    df = pd.read_csv(filename)
    if type =="regular" or type == "alpha":
        num = 11
    else:
        num = 10
    coord_cols = [f"coord{i}" for i in range(num)]
    sample = df.sample(n=samples)
    fig, ax = plt.subplots(figsize=(8, 6))
    #ax.set_aspect("equal")
    for index, row in sample.iterrows():
        points = [ast.literal_eval(row[c]) for c in coord_cols]
        xs, ys = zip(*points)
        if type == "regular":
            ax.plot(xs, ys, marker="o", label=f"F={row['F']:.3g}, E={row['E']:.3g}, base={row['base']:.3g}, height={row['height']:.3g}, L={row['length']:.3g}")
        if type == "alpha":
            ax.plot(xs, ys, marker="o", label=f"alpha={row['alpha']:.3g}")
        if type == "4eq":
            ax.plot(xs, ys, marker="o", label=f"F={row['F']:.3g}, E={row['E']:.3g}, base={row['base']:.3g}, height={row['height']:.3g}, L={row['length']:.3g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("4 random rows")
    plt.savefig(figname)

makeplot(FILE, FIGNAME, SAMPLES)
