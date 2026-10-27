import numpy as np
import matplotlib.pyplot as plt
from modell import (initial, voltage, sys, deposit, ellipse, POS, VEL, NOZZLE, FIL,
                    V_hold, V_max, T_dep, T_COOL, HOMING_U, SETTLE_TIME, SPEED_STOP,
                    WIPE_TIME, LOAD_TIME, PRINT_TIME, N, DT)

# setting for the simulation
REQUEST_TIME = 3.0
POS_TOL, SPEED_TOL = 1e-5, 1e-4
T_END, DONE_TIME, SAMPLE_EVERY = 140.0, 20.0, 40

# modes
AT_BOTTOM = ('Left', 'Wiping', 'Cooling', 'Loading', 'Heating')
AT_ORIGIN = ('Wiping', 'Cooling', 'Loading', 'Heating')


def arrived(z, target):
    return np.linalg.norm(z[POS] - target) < POS_TOL and np.linalg.norm(z[VEL]) < SPEED_TOL


def command(mode, z, clock, saved):
    if mode in ('Printing', 'Done'):
        return voltage(z, *ellipse(clock))
    if mode == 'Returning':
        return voltage(z, saved)
    if mode == 'Down':
        u = voltage(z, np.array([saved[0], 0]))
        u[1] = HOMING_U
        return u
    u = voltage(z, np.zeros(2))
    if mode == 'Left':
        u[0] = HOMING_U
    return u


# start with head at rest on the ellipse
z = initial([0.16, 0.10])
mode, timer, clock = 'Printing', 0.0, 0.0
saved, requested = z[POS].copy(), False
bottom = left = False
length, n = 0.0, 0
path = np.array([ellipse(t)[0] for t in np.linspace(0, PRINT_TIME, 1001)])
ds = np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / N
history, events = [], []

# main loop for sim
for step in range(int(T_END / DT)):
    t = step * DT
    old_mode = mode
    C = t >= REQUEST_TIME and not requested
    vx, vy = z[VEL]
    speed = np.linalg.norm(z[VEL])
    Tn = z[NOZZLE][0]

    if mode == 'Printing' and C:
        saved, requested = z[POS].copy(), True
        bottom = left = False
        mode = 'Down'
    elif mode == 'Down' and timer > SETTLE_TIME and abs(vy) < SPEED_STOP:
        bottom, mode = True, 'Left'
    elif mode == 'Left' and timer > SETTLE_TIME and abs(vx) < SPEED_STOP and bottom:
        left, mode = True, 'Wiping'
    elif mode == 'Wiping' and timer >= WIPE_TIME:
        mode = 'Cooling'
    elif mode == 'Cooling' and Tn <= T_COOL:
        mode = 'Loading'
    elif mode == 'Loading' and timer >= LOAD_TIME:
        mode = 'Heating'
    elif mode == 'Heating' and Tn >= T_dep:
        mode = 'Returning'
    elif mode == 'Returning' and arrived(z, saved):
        mode = 'Printing'
    elif mode == 'Printing' and clock >= PRINT_TIME and arrived(z, ellipse(PRINT_TIME)[0]):
        mode = 'Done'

    # mode change resets timer
    if mode != old_mode:
        timer = 0.0
        events.append((t, mode))

    # the inputs
    heater = {'Cooling': 0.0, 'Loading': 0.0, 'Heating': V_max}.get(mode, V_hold)
    u = command(mode, z, clock, saved)

    # printing
    printing = mode == 'Printing' and clock < PRINT_TIME
    if printing:
        n = deposit(z, n, length, ds)
        z[FIL + n - 1] = Tn

    if step % SAMPLE_EVERY == 0:
        history.append([t, *z[POS], Tn, heater, z[FIL:FIL + n].min(), z[FIL:FIL + n].max()])

    # looping filament ends
    z += DT * sys(z, u, heater, n, ds, closed=(mode == 'Done'))

    # head cant pass x = 0 or y = 0
    hit = z[POS] < 0
    z[POS][hit] = 0
    z[VEL][hit] = 0
    if mode in AT_BOTTOM:
        z[1], z[3] = 0, 0
    if mode in AT_ORIGIN:
        z[0], z[2] = 0, 0

    # timing
    if printing:
        length += speed * DT
        clock = min(PRINT_TIME, clock + DT)
    timer += DT
    if mode == 'Done' and timer >= DONE_TIME:
        break


h = np.array(history)
fig, ax = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
ax[0, 0].plot(*path.T, 'r:', label='Desired ellipse')
ax[0, 0].plot(h[:, 1], h[:, 2], color='yellow', label='Simulated path')
ax[0, 0].plot(*saved, 'ro', label='Saved position')
ax[0, 0].set(xlabel='x [m]', ylabel='y [m]', aspect='equal',
             title='Print, park, return and resume')
ax[0, 0].legend(fontsize=8)
ax[0, 1].plot(h[:, 0], h[:, 3], color='red');  ax[0, 1].set(xlabel='Time [s]', ylabel='Nozzle [C]')
ax[1, 0].plot(h[:, 0], h[:, 4], color='green');  ax[1, 0].set(xlabel='Time [s]', ylabel='Heater input [V]')
ax[1, 1].set_prop_cycle(color=['green', 'red'])
ax[1, 1].plot(h[:, 0], h[:, 5:7], label=['Minimum', 'Maximum'])
ax[1, 1].set(xlabel='Time [s]', ylabel='Deposited filament [C]')
ax[1, 1].legend()
for axis in (ax[0, 1], ax[1, 0], ax[1, 1]):
    for time, _ in events:
        axis.axvline(time, color='grey', alpha=0.25, linewidth=0.8)
fig.savefig('q3_1.png', dpi=160)
plt.show()