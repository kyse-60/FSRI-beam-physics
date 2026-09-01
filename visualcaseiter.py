import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("CaseIteration.csv")

yaxis = "MSE-loss"
# yaxis = "accuracy"
log_scaling = False

plt.plot(df["CASE"], df[yaxis])
if log_scaling:
    plt.xscale("log")
    plt.yscale("log")
plt.xlabel("Cases")
plt.ylabel(yaxis)
plt.title(f"{yaxis} vs training set size")
plt.savefig(f"{yaxis}vcasesNOlog.jpg")
plt.show()