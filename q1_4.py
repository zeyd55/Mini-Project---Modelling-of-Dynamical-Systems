import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# parameters we assumed
m = 0.30
r_m = 0.01
J_m = 2e-5
L_m = 0.01
R_m = 2.0
K_m = 0.05
mu_r = 0.02
c_x = 0.20
c_y = 0.40

J_eff = J_m + m * r_m**2 / 2
c_d = r_m**2 * (3 * mu_r + (c_x + c_y) / 4)
c_c = r_m**2 * (mu_r + (c_x - c_y) / 4)
alpha = np.array([c_d + c_c, c_d - c_c]) / J_eff
gain = K_m * r_m / (2 * J_eff)

centre = np.array([0.10, 0.10])
T_ellipse = 10.0
T_star = 20.0

# progresses smoothly from 0 to 1
def smooth(t, duration):
    s = np.clip(t / duration, 0, 1)

    progress = 10*s**3 - 15*s**4 + 6*s**5
    progress_rate = (30*s**2 - 60*s**3 + 30*s**4) / duration
    progress_acceleration = (60*s - 180*s**2 + 120*s**3) / duration**2
    progress_jerk = (60 - 360*s + 360*s**2) / duration**3

    return progress, progress_rate, progress_acceleration, progress_jerk

# return the position, velocity, acceleration, and jerk
def ellipse(t):
    s, ds, dds, ddds = smooth(t, T_ellipse)
    phi = 2 * np.pi * s
    phi_rate = 2 * np.pi * ds
    phi_acceleration = 2 * np.pi * dds
    phi_jerk = 2 * np.pi * ddds

    q = np.array([0.06 * np.cos(phi), 0.04 * np.sin(phi)])
    dq_dphi = np.array([-0.06 * np.sin(phi), 0.04 * np.cos(phi)])

    position = centre + q
    velocity = dq_dphi * phi_rate
    acceleration = -q * phi_rate**2 + dq_dphi * phi_acceleration
    jerk = (
        -dq_dphi * phi_rate**3
        - 3 * q * phi_rate * phi_acceleration
        + dq_dphi * phi_jerk
    )

    return position, velocity, acceleration, jerk

# star has 5 outer tips and 5 corners
vertices = []
for k in range(10):
    angle = np.pi / 2 + k * np.pi / 5

    if k % 2 == 0:
        radius = 0.06
    else:
        radius = 0.03

    vertex_x = centre[0] + radius * np.cos(angle)
    vertex_y = centre[1] + radius * np.sin(angle)
    vertices.append([vertex_x, vertex_y])

# repeat the first point so we get a closed star
vertices.append(vertices[0])
vertices = np.array(vertices)

def star(t):
    segment_duration = T_star / 10
    k = int(t / segment_duration)

    # use last edge
    if k > 9:
        k = 9

    segment_time = t - k * segment_duration
    s, ds, dds, ddds = smooth(segment_time, segment_duration)
    delta = vertices[k + 1] - vertices[k]

    position = vertices[k] + s * delta
    velocity = ds * delta
    acceleration = dds * delta
    jerk = ddds * delta

    return position, velocity, acceleration, jerk

# feedforward voltages from the model in q1.3 in the report
def motor_inputs(t, reference):
    position, velocity, acceleration, jerk = reference(t)

    current = (acceleration + alpha * velocity) / gain
    dcurrent = (jerk + alpha * acceleration) / gain
    axis_voltages = L_m*dcurrent + R_m*current + 2*K_m/r_m*velocity
    ux = axis_voltages[0]
    uy = axis_voltages[1]

    ul = (ux + uy) / 2
    ur = (ux - uy) / 2

    return ul, ur

# the six-state  model from q1.2 in the report
def model(t, state, reference):
    theta_l, theta_r, wl, wr, il, ir = state
    ul, ur = motor_inputs(t, reference)

    dtheta_l = wl
    dtheta_r = wr
    dwl = (K_m*il - c_d*wl - c_c*wr) / J_eff
    dwr = (K_m*ir - c_c*wl - c_d*wr) / J_eff
    di_l = (ul - R_m*il - K_m*wl) / L_m
    di_r = (ur - R_m*ir - K_m*wr) / L_m

    return [dtheta_l, dtheta_r, dwl, dwr, di_l, di_r]

for name, reference, duration in [
    ("Ellipse", ellipse, T_ellipse),
    ("Star", star, T_star),
]:
    initial_position = reference(0)[0]
    theta_l0 = (initial_position[0] + initial_position[1]) / r_m
    theta_r0 = (initial_position[0] - initial_position[1]) / r_m
    initial = [theta_l0, theta_r0, 0, 0, 0, 0]

    t = np.linspace(0, duration, 1001)
    sol = solve_ivp(
        model,
        [0, duration],
        initial,
        args=(reference,),
        t_eval=t,
        max_step=0.02,
        rtol=1e-6,
        atol=1e-8,
    )

    x = r_m / 2 * (sol.y[0] + sol.y[1])
    y = r_m / 2 * (sol.y[0] - sol.y[1])

    desired = []
    voltages = []
    for time in t:
        position, velocity, acceleration, jerk = reference(time)
        ul, ur = motor_inputs(time, reference)
        desired.append(position)
        voltages.append([ul, ur])

    desired = np.array(desired)
    voltages = np.array(voltages)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(x, y, color="yellow", linewidth=2, label="Model")
    ax[0].plot(desired[:, 0], desired[:, 1],
               color="red", linestyle=":", label="Reference")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[0].set_title(name)
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(t, voltages[:, 0], color="tab:green", label=r"$u_l$")
    ax[1].plot(t, voltages[:, 1], color="tab:red", label=r"$u_r$")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Voltage [V]")
    ax[1].set_title("Motor input voltages")
    ax[1].grid(True)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"q1_4_{name.lower()}.png", dpi=160)

plt.show()

