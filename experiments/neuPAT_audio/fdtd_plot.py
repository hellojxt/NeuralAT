import numpy as np
import sys
import matplotlib.pyplot as plt

data_dir = sys.argv[1]
nc_spec = np.load(f"{data_dir}/nc_spec.npy")
fdtd_spec = np.load(f"{data_dir}/fdtd_spec.npy")

nc_spec = np.log10((nc_spec + 10e-6) / 10e-6)
fdtd_spec = np.log10((fdtd_spec + 10e-6) / 10e-6)

nc_spec = nc_spec.reshape(-1)
fdtd_spec = fdtd_spec.reshape(-1)

plt.plot(nc_spec, label="nc")
plt.plot(fdtd_spec, label="fdtd")
plt.legend()
plt.show()
plt.close()


# plt.subplot(211)
# plt.imshow(nc_spec)
# plt.subplot(212)
# plt.imshow(fdtd_spec)
# plt.show()
# plt.close()
