"""Train an LSTM on the dynamic cantilever dataset -- minimal teaching version.

The task: given a tip-force history f(tau) and a damping value gamma, predict
the beam deflection eta(xi, tau) at every spatial station and time step.
Labels come from ``dynamic_beam_dataset.npz`` (run dynamic_generate_data.py
first). The model sees the force only up to the current time step, so it is
causal, and it is trained on labels alone -- no physics is used here.

Run:
    python dynamic_generate_data.py          (once, to make the dataset)
    python dynamic_train_lstm.py
"""

import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

# training settings
N_STEPS = 6000          # gradient steps
BATCH_SIZE = 16         # trajectories per step
LEARNING_RATE = 1.5e-3


def make_inputs(loads, damping, dt):
    """Stack the three LSTM input channels: [f, df/dtau, gamma].

    The force rate uses a BACKWARD difference (f[k] - f[k-1]) so that step k
    never sees the future. gamma is one number per case, repeated over time.
    Returns an array of shape (n_cases, n_time, 3).
    """
    n_time = loads.shape[1]
    rate = np.zeros_like(loads)
    rate[:, 1:] = (loads[:, 1:] - loads[:, :-1]) / dt
    gamma = np.repeat(damping[:, None], n_time, axis=1)
    return np.stack([loads, rate, gamma], axis=2)


class FieldLSTM(nn.Module):
    """Force history in, deflection field out.

    The LSTM reads the (n_time, 3) input sequence step by step and keeps the
    beam's "memory" in its hidden state. At every step a small dense head
    turns the hidden state into the deflection at all spatial stations. The
    output is the raw network value: the clamp condition eta(0) = 0 and the
    rest condition eta(tau=0) = 0 are NOT built in -- they must be learned.
    """

    def __init__(self, n_stations, n_hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(3, n_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, n_stations))

    def forward(self, sequences):
        hidden, _ = self.lstm(sequences)      # (n_cases, n_time, n_hidden)
        return self.head(hidden)              # (n_cases, n_time, n_stations)


def train(model, inputs_train, target_train, inputs_val, target_val):
    """Minibatch training; keep the weights that score best on validation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val, best_state = float("inf"), None
    train_curve, val_curve = [], []

    for step in range(1, N_STEPS + 1):
        if step == int(0.7 * N_STEPS):     # finish with 10x smaller steps,
            for group in optimizer.param_groups:   # otherwise the loss keeps
                group["lr"] = LEARNING_RATE / 10   # bouncing near the end
        batch = torch.randint(0, len(inputs_train), (BATCH_SIZE,))
        loss = ((model(inputs_train[batch]) - target_train[batch]) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # tames rare spikes
        optimizer.step()
        train_curve.append(loss.item())

        if step % 500 == 0:
            with torch.no_grad():
                val = ((model(inputs_val) - target_val) ** 2).mean().item()
            val_curve.append((step, val))
            if val < best_val:
                best_val = val
                best_state = copy.deepcopy(model.state_dict())
            print(f"step {step:5d}   train {loss.item():.3e}   val {val:.3e}")

    model.load_state_dict(best_state)
    return train_curve, val_curve


def plot_training(train_curve, val_curve):
    figure, ax = plt.subplots(figsize=(5.5, 3.6), constrained_layout=True)
    ax.semilogy(train_curve, lw=0.7, alpha=0.6, label="training loss")
    steps, values = zip(*val_curve)
    ax.semilogy(steps, values, "o-", label="validation loss")
    ax.set(xlabel="training step", ylabel="mean-square error")
    ax.legend()
    figure.savefig("lstm_training_history.png", dpi=150)
    plt.show()


def plot_prediction(tau, xi, load, true_field, predicted_field):
    """One unseen case: the load, the tip response, and the full field."""
    figure, (ax_f, ax_u) = plt.subplots(1, 2, figsize=(9, 3.2),
                                        constrained_layout=True)
    ax_f.plot(tau, load, color="black")
    ax_f.set(xlabel="time $\\tau$", ylabel="tip force $f(\\tau)$")
    ax_u.plot(tau, true_field[:, -1], color="black", label="ground truth")
    ax_u.plot(tau, predicted_field[:, -1], "--", label="LSTM prediction")
    ax_u.set(xlabel="time $\\tau$", ylabel="tip deflection $\\eta(1,\\tau)$")
    ax_u.legend()
    figure.savefig("lstm_test_prediction.png", dpi=150)
    plt.show()

    scale = np.abs(true_field).max()
    figure, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True,
                                constrained_layout=True)
    panels = [("ground truth", true_field, 1.0),
              ("LSTM prediction", predicted_field, 1.0),
              ("error", predicted_field - true_field, 0.05)]
    for ax, (title, values, limit) in zip(axes, panels):
        image = ax.pcolormesh(tau, xi, values.T / scale, cmap="RdBu_r",
                              vmin=-limit, vmax=limit)
        figure.colorbar(image, ax=ax)
        ax.set(ylabel="position $\\xi$", title=title)
    axes[-1].set(xlabel="time $\\tau$")
    figure.savefig("lstm_test_field.png", dpi=150)
    plt.show()


# 1. load the dataset and build the input/target arrays
data = np.load("dynamic_beam_dataset.npz")
tau, xi, dt = data["tau"], data["xi"], float(data["dt"])
loads, damping, fields = data["loads"], data["damping"], data["fields"]
n_cases = len(loads)
print(f"loaded {n_cases} cases: loads {loads.shape}, fields {fields.shape}")

inputs = make_inputs(loads, damping, dt)
field_scale = np.sqrt((fields ** 2).mean())        # so the loss is order one
targets = fields / field_scale

# 2. split by whole trajectory (70 / 15 / 15), so test loads are truly unseen
order = np.random.default_rng(seed=0).permutation(n_cases)
train_cases = order[:int(0.70 * n_cases)]
val_cases = order[int(0.70 * n_cases):int(0.85 * n_cases)]
test_cases = order[int(0.85 * n_cases):]

as_tensor = lambda array: torch.tensor(array, dtype=torch.float32)
inputs_train, target_train = as_tensor(inputs[train_cases]), as_tensor(targets[train_cases])
inputs_val, target_val = as_tensor(inputs[val_cases]), as_tensor(targets[val_cases])
inputs_test = as_tensor(inputs[test_cases])

# 3. build and train the model
torch.manual_seed(0)
model = FieldLSTM(n_stations=xi.size)
print(f"\ntraining on {len(train_cases)} labeled trajectories")
train_curve, val_curve = train(model, inputs_train, target_train,
                               inputs_val, target_val)

# 4. evaluate on the unseen test trajectories
with torch.no_grad():
    predicted = model(inputs_test).numpy() * field_scale
true = fields[test_cases]
error = np.linalg.norm(predicted - true) / np.linalg.norm(true)
print(f"\ntest field error: {100 * error:.3f}%  "
      f"({len(test_cases)} unseen force histories)")
print(f"learned-boundary check: max |eta| at clamp = "
      f"{np.abs(predicted[:, :, 0]).max():.2e}, at tau=0 = "
      f"{np.abs(predicted[:, 0, :]).max():.2e} "
      f"(an embedded model would give exactly 0)")

# 5. figures: training curve, then one unseen test case
plot_training(train_curve, val_curve)
case = 0
plot_prediction(tau, xi, loads[test_cases[case]], true[case], predicted[case])
