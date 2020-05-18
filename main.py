from gtrfile import GtrFile
import matplotlib.pyplot as plt
import numpy as np
import sys


def main():
    path = "records/test.gtr"

    if len(sys.argv) > 1:
        path = sys.argv[1]

    gtr = GtrFile(path)

    print(gtr)

    t, s = gtr.get_samples(2001, 1199)

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
