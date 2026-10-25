import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Properties form the tabel
rho = 1240.0
cp = 1800.0 
k = 0.13

# These are the assumed values
a = 0.0004
T_dep = 200.0
T_bed = 60.0
T_air = 20.0
h_air = 10.0
h_bed = 100.0
t_print = 10.0
N = 200
dt = 0.01
uniform_tol = 1.0

D = k / (rho * cp)
beta_air = 3 * h_air / (rho * cp * a)
beta_bed = h_bed / (rho * cp * a)
T_eq = (3*h_air*T_air + h_bed*T_bed) / (3*h_air + h_bed)

# This is the same printing motion as in the q1_4 script
path_time = np.linspace(0, t_print, 10001)
z = path_time / t_print
progress = 10*z**3 - 15*z**4 + 6*z**5
dprogress_dz = 30*z**2 - 60*z**3 + 30*z**4

phi = 2 * np.pi * progress
dphi = 2 * np.pi * dprogress_dz / t_print
dx_dphi = -0.06 * np.sin(phi)
dy_dphi = 0.04 * np.cos(phi)
speed = dphi * np.sqrt(dx_dphi**2 + dy_dphi**2)

path_length = cumulative_trapezoid(speed, path_time, initial=0)
length = path_length[-1]

def simulate(n_cells=N, step=dt):
    ds = length / n_cells
    stability_factor = step * (3*D/ds**2 + beta_air + beta_bed)
    if stability_factor >= 1:
        raise ValueError("Time step too large for this spatial grid.")

    # the nzzle at end of cell
    deposit_time = np.zeros(n_cells)
    for cell in range(n_cells):
        cell_end = (cell + 1) * ds
        deposit_time[cell] = np.interp(cell_end, path_length, path_time)

    deposit_time[0] = 0.0
    times = np.arange(0, 120 + step/2, step)

    # mark the cells that havent been printed 
    T = np.full(n_cells, np.nan)
    active = 0
    history = []

    for time in times:
        # add cells
        while active < n_cells and deposit_time[active] <= time + 1e-12:
            T[active] = T_dep
            active = active + 1

        loop_closed = time >= t_print - 1e-12
        history.append(T.copy())

        if loop_closed:
            temperature_spread = np.max(T) - np.min(T)
            if temperature_spread < uniform_tol:
                saved_times = times[:len(history)]
                saved_temperature = np.array(history)
                return saved_times, saved_temperature

        old_temperature = T[:active].copy()
        temperature_curvature = np.zeros(active)

        # conduction of cells that neighbour eachother
        for cell in range(active - 1):
            difference = old_temperature[cell + 1] - old_temperature[cell]
            curvature = difference / ds**2
            temperature_curvature[cell] += curvature
            temperature_curvature[cell + 1] -= curvature

        if loop_closed:
            # connecting the last cell to the first cell
            seam_difference = old_temperature[0] - old_temperature[-1]
            seam_curvature = seam_difference / ds**2
            temperature_curvature[-1] += seam_curvature
            temperature_curvature[0] -= seam_curvature
        else:
            # there is no flux at the start
            tip_difference = T_dep - old_temperature[-1]
            tip_curvature = 2 * tip_difference / ds**2
            temperature_curvature[-1] += tip_curvature

        # updates  printed cells using the temperature from the previous step
        for cell in range(active):
            conduction = D * temperature_curvature[cell]
            air_cooling = beta_air * (old_temperature[cell] - T_air)
            bed_cooling = beta_bed * (old_temperature[cell] - T_bed)
            temperature_rate = conduction - air_cooling - bed_cooling
            T[cell] = old_temperature[cell] + step * temperature_rate

    raise RuntimeError("Uniformity criterion not reached within 120 s.")

time, temperature = simulate()

T_min = []
T_max = []
for index in range(len(time)):
    current_temperature = temperature[index]
    minimum_temperature = np.nanmin(current_temperature)

    if time[index] < t_print:
        maximum_temperature = T_dep
    else:
        maximum_temperature = np.nanmax(current_temperature)

    T_min.append(minimum_temperature)
    T_max.append(maximum_temperature)

T_min = np.array(T_min)
T_max = np.array(T_max)

t_uniform = time[-1]

cmap = plt.get_cmap("coolwarm").copy()
cmap.set_bad("white")
fig, ax = plt.subplots(figsize=(7, 4))
image = ax.imshow(
    temperature.T, origin="lower", aspect="auto", cmap=cmap,
    extent=(0, time[-1], 0, length), vmin=T_air, vmax=T_dep,
)
ax.axvline(t_print, color="black", linestyle="--", label="Printing finished")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Distance along filament [m]")
ax.set_title("Temperature during printing and cooling")
ax.legend(loc="upper right")
fig.colorbar(image, ax=ax, label="Temperature [deg C]")
fig.tight_layout()
fig.savefig("q1_6_temperature.png", dpi=160)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(time, T_min, color="blue", label="Minimum temperature")
ax.plot(time, T_max, color="red", label="Maximum temperature")
ax.axhline(T_eq, color="grey", linestyle=":",
           label=f"Equilibrium: {T_eq:.1f} deg C")
ax.axvline(t_print, color="black", linestyle="--", label="Printing finished")
ax.axvline(t_uniform, color="green", linestyle="--", label="Almost uniform")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Temperature [deg C]")
ax.set_title("Filament cooling")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig("q1_6_cooling.png", dpi=160)

plt.show()

