import random
from glob import iglob
from itertools import count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

plt.style.use("seaborn-v0_8")
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\renewcommand{\familydefault}{\sfdefault}",
    }
)

SAMPLING_INTERVAL = 1
FILE_NAME = "TCLAB_data_periodic_inputs"
DATA_FOLDER = Path(__file__).parent.parent / "data/"
IGNORE_IDXS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}


def output_path(mode: str, counter: int) -> Path:
    return DATA_FOLDER / DATA_FOLDER / f"TCLab{mode}{counter:02d}.mat"


def shuffle_datasets(ratio=0, ignore=IGNORE_IDXS):
    idxs = [
        Path(s).stem.removeprefix(f"{FILE_NAME}_")
        for s in iglob(f"{FILE_NAME}_*.npz", root_dir=DATA_FOLDER / "raw")
    ]
    idxs = list(filter(lambda idx: int(idx) not in ignore, idxs))
    random.shuffle(idxs)
    split = round(len(idxs) * ratio)
    return idxs[:split], idxs[split:]


train_idxs, test_idxs = shuffle_datasets(ignore=IGNORE_IDXS)
datasets = {"Train": train_idxs, "Test": test_idxs}
for mode, idxs in datasets.items():
    if len(idxs) == 0:
        continue
    XTrajs = np.empty(len(idxs), dtype=object)
    UTrajs = np.empty(len(idxs), dtype=object)
    for loc, idx in enumerate(idxs):
        data = np.load(DATA_FOLDER / "raw" / f"{FILE_NAME}_{idx}.npz")

        dt = SAMPLING_INTERVAL * data["dt"]
        XFull = np.hstack([data["X"], data["Y"][:, -1:]])
        XTrajs[loc] = XFull[:, ::SAMPLING_INTERVAL]

        n_steps = XTrajs[loc].shape[1] - 1
        UFine = data["U"][:, : n_steps * SAMPLING_INTERVAL]
        UTrajs[loc] = UFine.reshape(UFine.shape[0], n_steps, SAMPLING_INTERVAL).mean(
            axis=2
        )

    # Save to mat file
    idx = next(c for c in count() if not output_path(mode, c).exists())
    savemat(
        output_path(mode, idx),
        {f"X{mode}": XTrajs, f"U{mode}": UTrajs, "Ts": dt},
        oned_as="column",
    )

    # Plot trajectories
    fig, axs = plt.subplots(
        4, 1, sharex=True, layout="constrained", num=f"{mode} Trajectories"
    )

    X = np.hstack(XTrajs.tolist())
    U = np.hstack(UTrajs.tolist())
    time = np.arange(0, X.shape[1]) * dt

    input_mask = np.full(len(time), True)
    input_mask[np.cumsum([x.shape[1] for x in XTrajs]) - 1] = False
    time_u = time[input_mask]

    axs[0].plot(time, X[0, :], color="tab:blue")
    axs[1].plot(time, X[1, :], color="tab:purple")
    axs[2].plot(time_u, U[0, :], color="tab:blue", drawstyle="steps-post")
    axs[3].plot(time_u, U[1, :], color="tab:purple", drawstyle="steps-post")

    for ax in axs:
        for t in time[~input_mask][:-1]:
            ax.axvline(t + dt, color="r", lw=0.5)

    axs[3].set_xlabel(r"Time [s]")
    axs[0].set_ylabel("Temp. Heater 1\n" r"$[^\circ$C]")
    axs[1].set_ylabel("Temp. Heater 2\n" r"$[^\circ$C]")
    axs[2].set_ylabel("Input Heater 1\n" r"[\%]")
    axs[3].set_ylabel("Input Heater 2\n" r"[\%]")

    for ax in axs.flatten():
        ax.set_xlim([time[0], time[-1]])

        ax.yaxis.grid(True)
        ax.xaxis.grid(False)

    fig.align_labels()
    plt.show(block=False)
    plt.pause(0.1)

plt.show()
