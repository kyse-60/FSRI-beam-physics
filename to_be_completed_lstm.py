"""EXERCISE: train an LSTM on the dynamic cantilever dataset.

This is ``dynamic_train_lstm.py`` with two parts removed. Your job is to
implement them, following the step-by-step instructions in each stub:

    PART 1 -- make_inputs():  build the LSTM input channels
    PART 2 -- FieldLSTM:      define the network

Everything else (loading, splitting, the training loop, evaluation, and
plotting) is complete, so the moment your two parts are correct the whole
script runs end to end. Do read the provided train() function -- you will
be asked to explain what every line does.

The task: given a tip-force history f(tau) and a damping value gamma, predict
the beam deflection eta(xi, tau) at every spatial station and time step.
Labels come from ``dynamic_beam_dataset.npz`` (run dynamic_generate_data.py
first). The model must be CAUSAL -- at time step k it may only use the force
up to step k -- and it is trained on labels alone, no physics.

How to check yourself (seeds are fixed, so results are reproducible):
    - validation loss at the end of training:  about 9e-4
    - test field error:                        about 2.8 %

Run:
    python dynamic_generate_data.py          (once, to make the dataset)
    python to_be_completed_lstm.py
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


# ======================================================================
# PART 1 -- build the LSTM inputs
# ======================================================================
def make_inputs(loads, damping, dt):
    """Stack the three LSTM input channels: [f, df/dtau, gamma].

    Arguments
        loads:   (n_cases, n_time) array, one force history per row
        damping: (n_cases,) array, one gamma value per case
        dt:      the time step, a float

    Must return an array of shape (n_cases, n_time, 3), where the last axis
    holds, in this order:
        channel 0: the force f itself
        channel 1: the force rate df/dtau
        channel 2: the damping gamma, repeated at every time step

    Step by step:
    1. Make an all-zero array ``rate`` with the same shape as ``loads``
       (np.zeros_like).
    2. Fill rate[:, 1:] with the BACKWARD difference
       (f[k] - f[k-1]) / dt  --  in array form:
       (loads[:, 1:] - loads[:, :-1]) / dt.
       Why backward and not centered? A centered difference uses f[k+1],
       which lies in the future -- that would break causality. rate[:, 0]
       stays 0, which is consistent with the soft start f(0) = f'(0) = 0.
    3. gamma is one number per case but the LSTM needs one number per time
       step, so repeat it along a new time axis:
       np.repeat(damping[:, None], n_time, axis=1) -> (n_cases, n_time).
    4. Stack the three (n_cases, n_time) arrays along a NEW last axis with
       np.stack([...], axis=2) and return the result.
    """
    # YOUR CODE HERE (about 4 lines)
    raise NotImplementedError("implement make_inputs (PART 1)")


# ======================================================================
# PART 2 -- define the network
# ======================================================================
class FieldLSTM(nn.Module):
    """Force history in, deflection field out.

    The idea: an LSTM reads the (n_time, 3) input sequence one step at a
    time and carries the beam's "memory" (how it is currently moving) in its
    hidden state. At every step a small dense head turns that hidden state
    into the deflection at all spatial stations. The output is the raw
    network value: the clamp condition eta(0, tau) = 0 and the rest
    condition eta(xi, 0) = 0 are NOT built in -- the network must learn
    them from the labels.

    Step by step:
    1. In __init__, first call super().__init__(), then create two layers:
       a. self.lstm = nn.LSTM(3, n_hidden, batch_first=True)
          -- 3 input channels, n_hidden hidden units. batch_first=True
          means tensors are laid out (batch, time, channels).
       b. self.head: use nn.Sequential to chain
          nn.Linear(n_hidden, n_hidden), nn.Tanh(),
          nn.Linear(n_hidden, n_stations)
          -- a tiny two-layer network applied at every time step.
    2. In forward(sequences):
       a. Run the LSTM:  hidden, _ = self.lstm(sequences)
          ``hidden`` has shape (n_cases, n_time, n_hidden) and contains the
          hidden state at EVERY step. The state at step k has only seen
          inputs 0..k, so the model is causal by construction.
          (The second return value, ignored here, is the final state.)
       b. Return self.head(hidden). A Linear layer acts on the LAST axis,
          so this maps every hidden state to the n_stations deflections of
          that instant: shape (n_cases, n_time, n_stations).
    """

    def __init__(self, n_stations, n_hidden=64):
        # YOUR CODE HERE (about 5 lines)
        raise NotImplementedError("implement FieldLSTM.__init__ (PART 2)")

    def forward(self, sequences):
        # YOUR CODE HERE (2 lines)
        raise NotImplementedError("implement FieldLSTM.forward (PART 2)")


# ======================================================================
# The training loop -- PROVIDED. Read it carefully: batch sampling, the
# zero_grad / backward / step order, gradient clipping, the learning-rate
# drop, and best-on-validation checkpointing are all standard tools you
# will reuse in every project.
# ======================================================================
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
            if val < best_val:             # the final step is not always the
                best_val = val             # best one -- keep a snapshot of
                best_state = copy.deepcopy(model.state_dict())  # the winner
            print(f"step {step:5d}   train {loss.item():.3e}   val {val:.3e}")

    model.load_state_dict(best_state)
    return train_curve, val_curve


# ======================================================================
# Everything below is complete -- no changes needed.
# ======================================================================
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
