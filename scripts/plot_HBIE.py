import numpy as np

import matplotlib.pyplot as plt

HBIE_rerr_lst = np.loadtxt("HBIE_rerr_lst.txt")
CBIE_rerr_lst = np.loadtxt("CBIE_rerr_lst.txt")


n = len(HBIE_rerr_lst)
xs = np.arange(n) * 0.01 + 100.6
plt.plot(xs, HBIE_rerr_lst, label="HBIE")
plt.plot(xs, CBIE_rerr_lst, label="CBIE")

plt.xlabel("Wavenumber")
plt.ylabel("Relative error")
plt.yscale("log")
plt.legend()

# set x-axis ticks
plt.xticks()
plt.savefig("HBIE_CBIE_rerr.png", dpi=300)
