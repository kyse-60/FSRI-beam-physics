import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("MSEperepoch.csv")

log_scaling = True

plt.plot(df["Epoch"], df["MSEtraining"], label="Training")
plt.plot(df["Epoch"], df["MSEvalidation"], label="Validation")
if log_scaling:
    plt.xscale("log")
    plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("MSE loss vs epoch")
plt.legend()
plt.savefig(f"MSEperepoch{'log' if log_scaling else 'NOlog'}.jpg")
plt.show()