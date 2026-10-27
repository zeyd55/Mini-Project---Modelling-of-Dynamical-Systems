import numpy as np
import matplotlib.pyplot as plt
from modell import (initial, voltage, sys, deposit, POS, VEL, NOZZLE, FIL,
                    rm, V_hold, SPEED_STOP, N, DT)

# sim settings
CORNER_TOL = 1e-6
T_END, AFTER_DONE, SAMPLE_EVERY = 25.0, 1.0, 40

# path of the square
corners = np.array([[0, 0], [0.10, 0], [0.10, 0.10], [0, 0.10], [0, 0]])
targets = corners[1:]
modes = ['Right', 'Up', 'Left', 'Down', 'Done']
ds = np.linalg.norm(np.diff(corners, axis=0), axis=1).sum() / N


def read_encoders(z):
    vx, vy = z[VEL]
    return np.array([vx + vy, vx - vy]) / rm


def velocity_from(omega):
    return rm / 2 * np.array([omega[0] + omega[1], omega[0] - omega[1]])


# head starts at rest
z = initial([0, 0])
estimate = np.zeros(2)
side, n, length, done_time = 0, 0, 0.0, None
history, corner_errors = [], []

# main loop for sim
for step in range(int(T_END / DT)):
    t = step * DT
    omega = read_encoders(z)
    v_meas = velocity_from(omega)
    target = targets[min(side, 3)]

    if (side < 4 and np.linalg.norm(estimate - target) < CORNER_TOL
            and np.linalg.norm(v_meas) < SPEED_STOP):
        corner_errors.append(np.linalg.norm(z[POS] - target))
        side += 1
        target = targets[min(side, 3)]
        if side == 4:
            done_time = t

    # input
    u = voltage(z, target, p=estimate, v=v_meas)

    # printing
    if side < 4:
        n = deposit(z, n, length, ds)
        z[FIL + n - 1] = z[NOZZLE][0]

    if step % SAMPLE_EVERY == 0:
        history.append([t, *z[POS], *omega, side])

    # loop filament ends
    z += DT * sys(z, u, V_hold, n, ds, closed=(side == 4))
    estimate += DT * v_meas
    if side < 4:
        length += np.linalg.norm(v_meas) * DT

    if done_time is not None and t - done_time >= AFTER_DONE:
        break


h = np.array(history)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
ax[0].plot(*corners.T, 'r:', label='Desired')
ax[0].plot(h[:, 1], h[:, 2], color='yellow', label='Simulated')
ax[0].set(xlabel='x [m]', ylabel='y [m]', aspect='equal', title='Square')
ax[0].legend()
ax[1].set_prop_cycle(color=['green', 'red'])
ax[1].plot(h[:, 0], h[:, 3:5], label=['Left motor', 'Right motor'])
ax[1].set(xlabel='Time [s]', ylabel='Angular velocity [rad/s]')
ax[1].legend()
ax[2].step(h[:, 0], h[:, 5], where='post', color='green')
ax[2].set(xlabel='Time [s]', yticks=range(5), yticklabels=modes, title='Discrete mode')
fig.savefig('q3_2.png', dpi=160)
plt.show()