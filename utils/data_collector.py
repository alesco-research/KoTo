import os
from itertools import count
from pathlib import Path

import numpy as np
from plotter import TCLabPlotter
from scipy.io import loadmat
from serial.tools import list_ports
from tclab import TCLab, clock

os.environ.pop("GTK_MODULES", None)

INCLUDE_COOLDOWN = False
INPUT_FILE = None
EXPORT_NAME = "TCLAB_data_periodic_inputs"
DATA_FOLDER = Path(__file__).parent.parent / "data/raw"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
TCLAB_USB_IDS = ["16D0:0613", "1A86:7523", "2341:8036", "2341:0043", "2A03"]


def _out_path(counter: int) -> Path:
    return DATA_FOLDER / f"{EXPORT_NAME}_{counter:02d}.npz"


def _get_ambient_temperature(measured: float) -> float:
    print(f"\nCurrently measured temperature: {measured:.2f} °C")
    print(
        "\033[31m"
        "NOTE: This only reflects the ambient temperature if the TCLab has not been "
        "used within the last 30 minutes."
        "\033[0m\n"
    )
    entered = input(f"Enter the ambient temperature in °C [{measured:.2f}]: ")
    try:
        temperature = np.float64(entered.replace(",", ".") or measured)
    except ValueError as err:
        raise ValueError(f"'{entered}' is not a number.") from err
    if temperature < -20 or temperature > 40:
        raise ValueError(f"Input temperature {temperature:.2f} °C is unrealistic.")
    print(f"Ambient temperature set to {temperature:.2f} °C")
    return temperature


# Periods [s], magnitudes [%] and phases [rad] of one heater input
Signal = tuple[np.ndarray, np.ndarray, np.ndarray]


def _signal_from_cells(cells: list[np.ndarray], traj: int, heater: int) -> Signal:
    periods, magnitudes, phases = (np.ravel(c[traj, heater]) for c in cells)
    if not len(periods) == len(magnitudes) == len(phases):
        raise ValueError(f"Cell ({traj}, {heater}) has mismatching lengths.")
    return periods, magnitudes, phases


def _load_input_parameters(mat_file: Path) -> list[list[Signal]]:
    mat = loadmat(mat_file)
    cells = [mat[k] for k in ("periods", "magnitudes", "phases")]
    if cells[0].shape[1] != 2 or any(c.shape != cells[0].shape for c in cells):
        raise ValueError(
            f"Expected cell arrays of size (trajectories x 2) in {mat_file}, "
            f"got {[c.shape for c in cells]}."
        )

    return [
        [_signal_from_cells(cells, traj, i) for i in range(2)]
        for traj in range(len(cells[0]))
    ]


def _generate_input_parameters(number_of_trajectories: int) -> list[list[Signal]]:
    periods = np.arange(10, 150, 15)  # [s]
    shape = (number_of_trajectories, 2, len(periods))

    switch_off = np.random.rand(number_of_trajectories, 1)
    phases = 2 * np.pi * np.random.random_sample(shape)
    magnitudes = np.random.random_sample(shape)
    magnitudes *= 120 / magnitudes.sum(axis=-1, keepdims=True)
    magnitudes[:, 0] *= switch_off < 0.8
    magnitudes[:, 1] *= switch_off > 0.2

    return [
        [(periods, magnitudes[traj, i], phases[traj, i]) for i in range(2)]
        for traj in range(number_of_trajectories)
    ]


def input_parameters(
    number_of_trajectories: int, mat_file: Path | None = None
) -> list[list[Signal]]:
    """Parameters of the periodic heater inputs, indexed by [trajectory][heater].

    The number of frequencies may differ between signals. If `mat_file` is given,
    the parameters are loaded from the (trajectories x 2) cell arrays 'periods',
    'magnitudes' and 'phases' in it and `number_of_trajectories` is ignored.
    """
    if mat_file is not None:
        return _load_input_parameters(mat_file)
    return _generate_input_parameters(number_of_trajectories)


def find_tclab():
    hits = [
        p.device
        for p in list_ports.comports()
        if any(k in p.hwid.upper() for k in TCLAB_USB_IDS)
    ]
    if len(hits) != 1:
        for p in list_ports.comports():
            print(p.device, "|", p.description, "|", p.hwid)
        raise RuntimeError(f"Expected exactly one TCLab, found {hits}")
    return hits[0]


with TCLab(find_tclab()) as lab, TCLabPlotter() as plotter:
    number_of_trajectories = 10

    dt = np.float64(1)  # [s]
    input_duration = 400  # [s]
    cooldown_timeout = 200  # [s]
    T_amb = _get_ambient_temperature((lab.T1 + lab.T2) / 2)  # [°C]
    times = np.atleast_2d(np.arange(0, input_duration + dt, dt))  # [s]

    heatup_rng = np.array([[45, 70], [35, 45]])  # [°C]
    cooldown_rng = T_amb + np.array([5, 10])  # [°C]

    # Generate inputs
    signals = input_parameters(number_of_trajectories, INPUT_FILE)
    number_of_trajectories = len(signals)
    inputs = np.empty((number_of_trajectories, 2, times.shape[1]))

    for traj, heaters in enumerate(signals):
        for heater, (periods, magnitude, phase) in enumerate(heaters):
            waves = 1 + np.cos(2 * np.pi * times / periods[:, None] + phase[:, None])
            inputs[traj, heater] = magnitude @ waves / 2

    inputs = np.clip(inputs, 0, 100)
    input_length = inputs.shape[-1]

    for i in np.arange(number_of_trajectories):
        input1, input2 = inputs[i]

        max_temp = heatup_rng[:, [0]] + np.diff(heatup_rng) * np.random.rand(2, 1)
        cool_target = (cooldown_rng[0] + np.diff(cooldown_rng) * np.random.rand(1))[0]

        print(f"\nTrajectory {i + 1}/{number_of_trajectories}")
        print(f"\tHeat-up target: {max_temp.ravel()} °C")
        print(f"\tCool-down target: {cool_target:.2f} °C")
        print("\t[Heat-up] starting ...")

        heating = True
        X_cols, U_cols, t_cols = ([], [], [])
        for t in clock(np.inf, dt):
            temp1, temp2 = lab.T1, lab.T2
            idx = round(t / dt)

            if heating and idx >= input_length:
                heating = False
                cooldown_start = t
                print("\t[Heat-up] end of input reached\n\t[Cool-down] starting ...")

            if heating:
                q1, q2 = float(input1[idx]), float(input2[idx])

                stop_cond = max_temp < np.array([[temp1], [temp2]])
                if np.any(stop_cond):
                    heating = False
                    cooldown_start = t
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
                if t - cooldown_start >= cooldown_timeout:
                    print(
                        f"\t[Cool-down] timeout after {cooldown_timeout} s with "
                        f"T1={temp1:.2f} and T2={temp2:.2f} °C."
                    )
                    break

            lab.Q1(q1), lab.Q2(q2)
            lab.LED(100 * (round(t) % 2))

            if heating or INCLUDE_COOLDOWN:
                t_cols.append(t)
                U_cols.append([q1, q2])
                X_cols.append([temp1, temp2])
            plotter.update(temp1, temp2, q1, q2)

        if len(t_cols) < 2:
            print("\tWARNING: Too few samples recorded, trajectory discarded.")
            continue

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
