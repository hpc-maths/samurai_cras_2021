import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
# import seaborn.apionly as sns
import seaborn as sns
from cycler import cycler

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "font.sans-serif": ["Helvetica"]})

plt.style.use('seaborn-colorblind')

def set_size(width, myratio, fraction=1):
    """ Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * myratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

width = 452.9679
fig1 = plt.figure(figsize=set_size(width, 1.2))

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

pathname = './build/D2Q9_jump_compte_rendus'
x_reference_begin  = np.loadtxt(f'{pathname}/x_reference_begin.dat')
y_reference_begin  = np.loadtxt(f'{pathname}/y_reference_begin.dat')
y_exact_begin      = np.loadtxt(f'{pathname}/y_exact_begin.dat')
x_jump_begin       = np.loadtxt(f'{pathname}/x_jump_begin.dat')
y_jump_begin       = np.loadtxt(f'{pathname}/y_jump_begin.dat')

ax1.plot(x_reference_begin, y_exact_begin, '-', label="Analytical sol.")
ax1.plot(x_reference_begin, y_reference_begin, '--', label="Reference sol.")
ax1.plot(x_jump_begin, y_jump_begin, '.', fillstyle = 'none', label = "Jump sol.")

ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1.set_ylabel("$\\delta \\rho (t, x, y=0)$")
ax1.set_xlabel("$x$")
ax1.legend(fontsize = 6)

y_reference_final  = np.loadtxt(f'{pathname}/y_reference_final.dat')
y_exact_final      = np.loadtxt(f'{pathname}/y_exact_final.dat')
y_jump_final       = np.loadtxt(f'{pathname}/y_jump_final.dat')

ax2.plot(x_reference_begin, y_exact_final, '-')
ax2.plot(x_reference_begin, y_reference_final, '--')
ax2.plot(x_jump_begin, y_jump_final, '.', fillstyle = 'none')

ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.set_xlabel("$x$")
# ax2.set_ylabel("$\\rho'$")

plt.tight_layout()
plt.show()

print(np.max(y_exact_final) - np.min(y_exact_final))
