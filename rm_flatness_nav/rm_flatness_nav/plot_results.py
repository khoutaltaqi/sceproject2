import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def main():
    root = (
        Path.home()
        / 'robomaster_ws/src/rm_flatness_nav'
        / 'logs/assessment2'
    )

    runs = sorted(
        root.glob('run_*/execution.csv')
    )

    if not runs:
        raise FileNotFoundError(
            'Belum ada execution.csv'
        )

    source = runs[-1]

    with source.open() as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise ValueError('Log kosong')

    t = [
        float(row['t_s'])
        for row in rows
    ]

    fig, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.plot(
        t,
        [
            float(row['v_cmd_mps'])
            for row in rows
        ],
        label='v command',
        color='tab:blue',
    )

    axis.plot(
        t,
        [
            float(row['v_measured_mps'])
            for row in rows
        ],
        label='v measured',
        color='tab:green',
    )

    axis.set_xlabel('Time (s)')
    axis.set_ylabel('Forward speed (m/s)')
    axis.grid(True, alpha=0.3)

    volts = axis.twinx()

    volts.plot(
        t,
        [
            float(row['V1_V'])
            for row in rows
        ],
        label='V1 model',
        color='tab:red',
    )

    volts.set_ylabel(
        'Estimated motor voltage (V)'
    )

    lines1, labels1 = (
        axis.get_legend_handles_labels()
    )
    lines2, labels2 = (
        volts.get_legend_handles_labels()
    )

    axis.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='best',
    )

    fig.tight_layout()

    destination = (
        source.parent / 'execution.png'
    )

    fig.savefig(destination, dpi=180)
    plt.close(fig)

    print(destination)
