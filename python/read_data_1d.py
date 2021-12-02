import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 5,
    "font.sans-serif": ["Helvetica"]})

plt.style.use('seaborn-colorblind')

mksize = 1

f, axs = plt.subplots(3, 2, sharey=False)

pathname = './D1Q3_jump_compte_rendus'
ite2plot = 1599

mesh = h5py.File(f'{pathname}/MR_ite-0.h5', 'r')['mesh']
points = mesh['points']
connectivity = mesh['connectivity']

segments = np.zeros((connectivity.shape[0], 2, 2))
segments[:, :, 0] = points[:][connectivity[:]][:, :, 0]

data = mesh['fields']['u'][:]
centers = .5*(segments[:, 0, 0] + segments[:, 1, 0])
segments[:, :, 1] = data[:, np.newaxis]

axs[0, 0].plot(centers, data, '-', color = 'C2', linewidth = 1, alpha=0.5)
axs[1, 0].plot(centers, data, '-', color = 'C2', linewidth = 1, alpha=0.5)
axs[2, 0].plot(centers, data, '-', color = 'C2', linewidth = 1, alpha=0.5)
# axs[0, 1].plot(centers, data, '-', color = 'C2', linewidth = 1, alpha=0.5)
# axs[1, 1].plot(centers, data, '-', color = 'C2', linewidth = 1, alpha=0.5)
# axs[2, 1].plot(centers, data, '-', color = 'C2', linewidth = 1, alpha=0.5)

mesh = h5py.File(f'{pathname}/MR_ite-{ite2plot}.h5', 'r')['mesh']
points = mesh['points']
connectivity = mesh['connectivity']

segments = np.zeros((connectivity.shape[0], 2, 2))
segments[:, :, 0] = points[:][connectivity[:]][:, :, 0]

data = mesh['fields']['u'][:]
centers = .5*(segments[:, 0, 0] + segments[:, 1, 0])
segments[:, :, 1] = data[:, np.newaxis]

index = np.argsort(centers)

axs[0, 0].plot(centers[index[:2048]], data[index[:2048]], '.', markersize = mksize, color = 'C1')
axs[0, 0].plot(centers[index[2048:]], data[index[2048:]], '.', markersize = mksize, color = 'C0')
axs[0, 1].plot(centers, data, '.', markersize = mksize, color = 'C1')

axs[0, 1].set_xlim([1.6, 1.9])
visible_y = data[np.logical_and(1.6 < centers, centers < 1.9)]
if len(visible_y):
    axs[0, 1].set_ylim(np.min(visible_y)*1.1, np.max(visible_y)*1.1)
axs[0, 0].set_ylabel("$\\overline{u}(T)$")
axs[0, 0].set_title("$\\textrm{Multiresolution}$")
axs[0, 1].set_title("$\\textrm{Multiresolution (Zoom)}$")

axs[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

mesh = h5py.File(f'{pathname}/LW_ite-{ite2plot}.h5', 'r')['mesh']
points = mesh['points']
connectivity = mesh['connectivity']

segments = np.zeros((connectivity.shape[0], 2, 2))
segments[:, :, 0] = points[:][connectivity[:]][:, :, 0]

data = mesh['fields']['u'][:]
centers = .5*(segments[:, 0, 0] + segments[:, 1, 0])
segments[:, :, 1] = data[:, np.newaxis]

axs[1, 0].plot(centers[index[:2048]], data[index[:2048]], '.', markersize = mksize, color = 'C1')
axs[1, 0].plot(centers[index[2048:]], data[index[2048:]], '.', markersize = mksize, color = 'C0')
axs[1, 1].plot(centers, data, '.', markersize = mksize, color = 'C1')
axs[1, 1].set_xlim([1.6, 1.9])
visible_y = data[np.logical_and(1.6 < centers, centers < 1.9)]
if len(visible_y):
    axs[1, 1].set_ylim(np.min(visible_y)*1.1, np.max(visible_y)*1.1)
axs[1, 0].set_ylabel("$\\overline{u}(T)$")
axs[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[1, 0].set_title("$\\textrm{Lax-Wendroff: Fakhari and Lee}$")
axs[1, 1].set_title("$\\textrm{Lax-Wendroff: Fakhari and Lee (Zoom)}$")


# mesh = h5py.File('./rohde/LBM_D1Q3_acoustic_wave_mesh_jump_Rohde_-199.h5', 'r')['mesh']
mesh = h5py.File(f'{pathname}/Rohde_ite-{ite2plot}.h5', 'r')['mesh']
points = mesh['points']
connectivity = mesh['connectivity']

segments = np.zeros((connectivity.shape[0], 2, 2))
segments[:, :, 0] = points[:][connectivity[:]][:, :, 0]

data = mesh['fields']['u'][:]
centers = .5*(segments[:, 0, 0] + segments[:, 1, 0])
segments[:, :, 1] = data[:, np.newaxis]

axs[2, 0].plot(centers[index[:2048]], data[index[:2048]], '.', markersize = mksize, color = 'C1')
axs[2, 0].plot(centers[index[2048:]], data[index[2048:]], '.', markersize = mksize, color = 'C0')
axs[2, 1].plot(centers, data, '.', markersize = mksize, color = 'C1')
axs[2, 1].set_xlim([1.6, 1.9])
visible_y = data[np.logical_and(1.6 < centers, centers < 1.9)]
if len(visible_y):
    axs[2, 1].set_ylim(np.min(visible_y)*1.1, np.max(visible_y)*1.1)
axs[2, 1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

axs[2, 0].set_title("$\\textrm{Rohde et al.}$")
axs[2, 1].set_title("$\\textrm{Rohde et al. (Zoom)}$")

axs[2, 0].set_ylabel("$\\overline{u}(T)$")
axs[2, 0].set_xlabel("$x$")
axs[2, 1].set_xlabel("$x$")

plt.tight_layout()

plt.show()