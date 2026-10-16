import os
from itertools import count
from pathlib import Path

import numpy as np
from plotter import TCLabPlotter
from tclab import TCLab, clock

os.environ.pop("GTK_MODULES", None)

EXPORT_NAME = "TCLAB_data_periodic_inputs"
DATA_FOLDER = Path(__file__).parent.parent / "data/raw"


def _out_path(counter: int) -> Path:
    return DATA_FOLDER / f"{EXPORT_NAME}_{counter:02d}.npz"


with TCLab() as lab, TCLabPlotter() as plotter:
    print(f"Temperature at start: {(lab.T1 + lab.T2) / 2}")

    dt = np.float64(1)  # [s]
    max_time = 1000  # [s]
    T_amb = np.float64(20)  # [°C]
    number_of_trajectories = 10
    times = np.atleast_2d(np.arange(0, max_time + dt, dt))  # [s]
    heatup_rng = np.array([45, 70])  # [°C]
    cooldown_rng = np.array([30, 40])  # [°C]
    periods = np.arange(10, 150, 15).reshape(-1, 1)  # [s]

    np.random.seed(42)
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    for i in np.arange(number_of_trajectories):
        phases = 2 * np.pi * np.random.random_sample((periods.shape[0], 2))
        magnitudes = np.random.random_sample(phases.shape)
        magnitudes /= 2 * magnitudes.sum(axis=0, keepdims=True)
        switch_off = np.random.rand(1)

        input1 = magnitudes[:, 0] @ np.cos(2 * np.pi * times / periods + phases[:, [0]])
        input2 = magnitudes[:, 1] @ np.cos(2 * np.pi * times / periods + phases[:, [1]])
        input1 = (switch_off < 0.8) * np.clip((120 * (input1 + 0.5)), 0, 100).flatten()
        input2 = (switch_off > 0.2) * np.clip((120 * (input2 + 0.5)), 0, 100).flatten()
        input_length = len(input1)

        max_temp = (heatup_rng[0] + np.diff(heatup_rng) * np.random.rand(1))[0]
        cool_target = (cooldown_rng[0] + np.diff(cooldown_rng) * np.random.rand(1))[0]

        print(f"\nTrajectory {i + 1}/{number_of_trajectories}")
        print(f"\tHeat-up target: {max_temp:.2f} °C")
        print(f"\tCool-down target: {cool_target:.2f} °C")

        print("\t[Heat-up] starting ...")
        led_on = True
        X_cols, U_cols = ([], [])
        for th in clock(max_time, dt):
            idx = int(th / dt) % input_length
            q1 = float(input1[idx])
            q2 = float(input2[idx])
            lab.Q1(q1), lab.Q2(q2)

            temp1, temp2 = lab.T1, lab.T2
            U_cols.append([q1, q2])
            X_cols.append([temp1, temp2])
            plotter.update(temp1, temp2, q1, q2)

            lab.LED(100 * led_on)
            led_on = not led_on

            if max_temp <= max(temp1, temp2):
                print(f"\t[Heat-up] target {max_temp:.2f} °C reached.")
                break

        print("\t[Cool-down] starting ...")
        lab.Q1(0), lab.Q2(0)
        for _ in clock(max_time, dt):
            temp1, temp2 = lab.T1, lab.T2
            U_cols.append([0.0, 0.0])
            X_cols.append([temp1, temp2])
            plotter.update(temp1, temp2, q1, q2)

            lab.LED(100 * led_on)
            led_on = not led_on

            if max(temp1, temp2) < cool_target:
                print(f"\t[Cool-down] target {cool_target:.2f} °C reached.")
                break

        X = np.array(X_cols, dtype=np.float64).T
        U = np.array(U_cols, dtype=np.float64).T

        idx = next(c for c in count() if not _out_path(c).exists())
        out_path = _out_path(idx)
        np.savez(out_path, Y=X[:, 1:], X=X[:, :-1], U=U[:, :-1], T_amb=T_amb, dt=dt)
        print(f"\tSaved {out_path}.")

    lab.Q1(0), lab.Q2(0)
    print("\nDone. Heaters off, connection closed.")
