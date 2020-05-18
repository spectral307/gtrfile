from pandas import to_datetime
from gtrfile import GtrFile
import matplotlib.pyplot as plt
import numpy as np
import sys


def calc_time_in_seconds(duration, rate):
    return np.arange(duration, step=1/rate)


def calc_time_index(duration, rate):
    return to_datetime(calc_time_in_seconds(duration, rate),
                       unit="s")


def main():
    path = "test_record.gtr"

    if len(sys.argv) > 1:
        path = sys.argv[1]

    gtr = GtrFile(path)

    print(gtr)

    s = np.empty(gtr.samples_number_per_input, dtype=gtr.dtype)
    gtr.get_samples(0, s)

    t = calc_time_index(gtr.header["time"], gtr.header["rate"])

    ax = plt.subplot()
    plt.plot(t, s[gtr.header["inputs"][0]["name"]],
             label=gtr.header["inputs"][0]["name"])
    plt.plot(t, s[gtr.header["inputs"][1]["name"]],
             label=gtr.header["inputs"][1]["name"])
    ax.grid()
    ax.autoscale(enable=True, axis="x", tight=True)
    ax.legend()
    ax.set_xlabel("time, s")

    plt.show()


if __name__ == "__main__":
    main()
