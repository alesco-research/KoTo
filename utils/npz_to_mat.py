from pathlib import Path

import numpy as np
from scipy.io import savemat

data_folder = Path(__name__).parent.parent / "data/"
reference_data = np.load(data_folder / "raw/TCLab_reference.npz")

datasets = {"Train": [1, 2, 3, 4, 5], "Test": [7, 6, 8]}
for key, value in datasets.items():
    XTrajs = np.empty(len(value), dtype=object)
    UTrajs = np.empty(len(value), dtype=object)
    for i, j in enumerate(value):
        data = np.load(data_folder / f"raw/TCLab_data_u0_EDMDc_{j}.npz")

        T_amb = np.full((1, data["U"].shape[1]), data["T_amb"])

        XTrajs[i] = np.hstack([data["X"], data["Y"][:, -1:]]) + data["T_amb"]
        UTrajs[i] = data["U"][:, 0:-1]  # np.vstack([T_amb, data["U"]])[:, 0:-1]

    savemat(
        data_folder / f"TCLab{key}.mat",
        {f"X{key}": XTrajs, f"U{key}": UTrajs, "Ts": reference_data["dt"]},
        oned_as="column",
    )
