import os
from itertools import count
from pathlib import Path

import numpy as np
from plotter import TCLabPlotter
from tclab import TCLab, clock

os.environ.pop("GTK_MODULES", None)

EXPORT_NAME = "TCLAB_data_periodic_inputs"
DATA_FOLDER = Path(__file__).parent.parent / "data/raw"
INCLUDE_COOLDOWN = False


def _out_path(counter: int) -> Path:
    return DATA_FOLDER / f"{EXPORT_NAME}_{counter:02d}.npz"


with TCLab() as lab, TCLabPlotter() as plotter:
    print(f"Temperature at start: {(lab.T1 + lab.T2) / 2}")
    number_of_trajectories = 10

    dt = np.float64(1)  # [s]
    max_time = 1000  # [s]
    T_amb = np.float64(22)  # [°C]
    times = np.atleast_2d(np.arange(0, max_time + dt, dt))  # [s]

    heatup_rng = np.array([[45, 70], [35, 45]])  # [°C]
    cooldown_rng = T_amb + np.array([5, 10])  # [°C]
    periods = np.arange(10, 150, 15).reshape(-1, 1)  # [s]

    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    for i in np.arange(number_of_trajectories):
        phases = 2 * np.pi * np.random.rand(periods.shape[0], 2)
        magnitudes = np.random.random_sample(phases.shape)
        magnitudes /= 2 * magnitudes.sum(axis=0, keepdims=True)
        switch_off = np.random.rand(1)

        input1 = magnitudes[:, 0] @ np.cos(2 * np.pi * times / periods + phases[:, [0]])
        input2 = magnitudes[:, 1] @ np.cos(2 * np.pi * times / periods + phases[:, [1]])
        input1 = (switch_off < 0.8) * np.clip((140 * (input1 + 0.5)), 0, 100).flatten()
        input2 = (switch_off > 0.2) * np.clip((140 * (input2 + 0.5)), 0, 100).flatten()
        input_length = len(input1)

        max_temp = heatup_rng[:, [0]] + np.diff(heatup_rng) * np.random.rand(2, 1)
        cool_target = (cooldown_rng[0] + np.diff(cooldown_rng) * np.random.rand(1))[0]

        print(f"\nTrajectory {i + 1}/{number_of_trajectories}")
        print(f"\tHeat-up target: {max_temp.ravel()} °C")
        print(f"\tCool-down target: {cool_target:.2f} °C")
        print("\t[Heat-up] starting ...")

        heating = True
        X_cols, U_cols, t_cols = ([], [], [])
        for t in clock(max_time, dt):
            temp1, temp2 = lab.T1, lab.T2

            if heating:
                idx = round(t / dt) % input_length
                q1, q2 = float(input1[idx]), float(input2[idx])

                stop_cond = max_temp < np.array([[temp1], [temp2]])
                if np.any(stop_cond):
                    heating = False
                    print(
                        f"\t[Heat-up] target reached with T1={temp1:.2f} and "
                        f"T2={temp2:.2f} °C\n"
                        "\t[Cool-down] starting ..."
                    )
            else:
                q1, q2 = 0, 0
                if max(temp1, temp2) < cool_target:
                    print("\t[Cool-down] target reached.")
                    break

            lab.Q1(q1), lab.Q2(q2)
            lab.LED(100 * (round(t) % 2))

            if heating or INCLUDE_COOLDOWN:
                t_cols.append(t)
                U_cols.append([q1, q2])
                X_cols.append([temp1, temp2])
            plotter.update(temp1, temp2, q1, q2)

        X = np.array(X_cols, dtype=np.float64).T
        U = np.array(U_cols, dtype=np.float64).T
        T = np.array(t_cols, dtype=np.float64)

        dt_err = np.abs(np.diff(T) - dt)
        if np.any(dt_err > 0.1 * dt):
            print(
                f"\tWARNING: {np.sum(dt_err > 0.1 * dt)} non-uniform sample(s), "
                f"max deviation {dt_err.max():.3f} s"
            )

        idx = next(c for c in count() if not _out_path(c).exists())
        out_path = _out_path(idx)
        np.savez(
            out_path, t=T, Y=X[:, 1:], X=X[:, :-1], U=U[:, :-1], T_amb=T_amb, dt=dt
        )
        print(f"\tSaved {out_path}.")

    lab.Q1(0), lab.Q2(0)
    print("\nDone. Heaters off, connection closed.")
