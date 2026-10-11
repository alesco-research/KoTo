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

SAMPLING_INTERVAL = 3
data_folder = Path(__file__).parent.parent / "data/"
reference_data = np.load(data_folder / "raw/TCLab_reference.npz")
dt = SAMPLING_INTERVAL * reference_data["dt"]

datasets = {"Train": [1, 2, 3, 4, 5], "Test": [7, 6, 8]}
for mode, idxs in datasets.items():
    XTrajs = np.empty(len(idxs), dtype=object)
    UTrajs = np.empty(len(idxs), dtype=object)
    for loc, idx in enumerate(idxs):
        data = np.load(data_folder / f"raw/TCLab_data_u0_EDMDc_{idx}.npz")

        XFull = np.hstack([data["X"], data["Y"][:, -1:]]) + data["T_amb"]
        XTrajs[loc] = XFull[:, ::SAMPLING_INTERVAL]

        n_steps = XTrajs[loc].shape[1] - 1
        UFine = data["U"][:, : n_steps * SAMPLING_INTERVAL]
        UTrajs[loc] = UFine.reshape(UFine.shape[0], n_steps, SAMPLING_INTERVAL).mean(
            axis=2
        )

    # Save to mat file
    savemat(
        data_folder / f"TCLab{mode}1.mat",
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
