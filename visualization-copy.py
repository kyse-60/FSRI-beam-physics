import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

FILE = "output4eq-time-6.9866809000086505s.csv"
type = "4eq" 
FIGNAME = f'VISUAL{type}'
SAMPLES = 10

def alpha(row):
    I = (row['base'] * row['height'] ** 3) / 12
    return (row['F'] * row['length'] ** 2) / (2 * row['E'] * I)

def makeplot(filename, figname, samples):
    df = pd.read_csv(filename)
    num = sum(c.startswith('coord') for c in df.columns)
    df['lambda'] = df.apply(alpha, axis=1)
    unique = df.drop_duplicates('lambda')
    targets = np.linspace(unique['lambda'].min(), unique['lambda'].max(), samples)
    chosen = []
    for t in targets:
        i = (unique['lambda'] - t).abs().drop(chosen).idxmin()
        chosen.append(i)
    sample = unique.loc[chosen].sort_values('lambda')
    xi = np.arange(num)/(num- 1)

    fig1, ax = plt.subplots(figsize=(8, 6))
    fig2, axes = plt.subplots(2, 2, figsize=(11, 8))

    for index, row in sample.iterrows():
        L = row['length']
        points = [ast.literal_eval(row[f"coord{i}"]) for i in range(num)]
        xs, ys = zip(*points)
        X = np.array(xs) / L
        Y = -np.array(ys) / L
        phi = np.degrees([row[f"phi{i}"] for i in range(num)])
        K = np.array([row[f"k{i}"] for i in range(num)]) * L

        ax.plot(X, Y, label=fr"$\lambda$ = {alpha(row):.3g}")
        axes[0, 0].plot(xi, X - xi)
        axes[0, 1].plot(xi, Y)
        axes[1, 0].plot(xi, phi)
        axes[1, 1].plot(xi, K)

    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("deflected shapes")
    fig1.savefig(figname + "_shapes.jpg")

    axes[0, 0].set_title(r"$X - \xi$ (foreshortening)")
    axes[0, 1].set_title("$Y$ (deflection)")
    axes[1, 0].set_title(r"$\phi$ [deg] (rotation)")
    axes[1, 1].set_title("$K$ (curvature)")
    for a in axes.flat:
        a.set_xlabel(r"arclength $\xi$")
    fig2.tight_layout()
    fig2.savefig(figname + "_fields.jpg")
    plt.close('all')

makeplot(FILE, FIGNAME, SAMPLES)