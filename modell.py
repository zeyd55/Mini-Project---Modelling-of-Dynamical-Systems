import numpy as np

# parameters from report
m, rm, Jm = 0.30, 0.01, 2e-5
Rm, Lm, Km = 2.0, 0.01, 0.05
mu_r, cx, cy = 0.02, 0.20, 0.40

Jeff = Jm + m * rm**2 / 2
cd = rm**2 * (3*mu_r + (cx + cy)/4)
cc = rm**2 * (mu_r + (cx - cy)/4)
alpha = np.array([cd + cc, cd - cc]) / Jeff
g = Km * rm / (2 * Jeff)

# nozzle parameters 
Ln, Cn, Rn = 0.01, 0.001, 12.0
Ct, Rt, Te = 5.0, 6.0, 20.0
T_dep = 200.0
V_hold = np.sqrt(Rn * (T_dep - Te) / Rt)
V_max = 24.0

# filament parameters
rho, cp, k, a = 1240, 1800, 0.13, 0.4e-3
Tb, h_air, h_bed = 60.0, 10.0, 100.0
N, DT = 40, 0.0005

# printing time
PRINT_TIME = 10.0

# thresholds used
HOMING_U = -0.30
SETTLE_TIME = 0.2
SPEED_STOP = 1e-5
WIPE_TIME, LOAD_TIME = 1.0, 2.0
T_COOL = 60.0

# layout of state
POS, VEL, CUR, NOZZLE = slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 9)
FIL = 9


# head starts at rest
def initial(position):
    z = np.zeros(FIL + N)
    z[POS] = position
    z[NOZZLE] = [T_dep, V_hold, V_hold / Rn]
    z[FIL:] = Te
    return z


def voltage(z, ref_p, ref_v=0, ref_a=0, p=None, v=None, kp=100, kd=20):
    p = z[POS] if p is None else p
    v = z[VEL] if v is None else v
    force = ref_a + alpha*ref_v + kp*(ref_p - p) + kd*(ref_v - v)
    return Rm/g * force + 2*Km/rm * v


def sys(z, u, heater, n, ds, closed=False):
    dz = np.zeros_like(z)

    # motors
    dz[POS] = z[VEL]
    dz[VEL] = -alpha*z[VEL] + g*z[CUR]
    dz[CUR] = (u - Rm*z[CUR] - 2*Km/rm*z[VEL]) / Lm

    # nozzle
    Tn, vn, i_n = z[NOZZLE]
    dz[NOZZLE] = [(vn**2/Rn - (Tn - Te)/Rt) / Ct,
                  (i_n - vn/Rn) / Cn,
                  (heater - vn) / Ln]

    # filament
    if n:
        T = z[FIL:FIL + n]
        padded = np.pad(T, 1, mode="wrap" if closed else "edge")
        d2T = (padded[:-2] - 2*T + padded[2:]) / ds**2
        loss = (3*h_air*(T - Te) + h_bed*(T - Tb)) / a
        dz[FIL:FIL + n] = (k*d2T - loss) / (rho*cp)
    return dz

# printing
def deposit(z, n, length, ds):
    new_n = min(N, int(length/ds) + 1)
    z[FIL + n:FIL + new_n] = z[6]
    return new_n


def ellipse(t, center=(0.10, 0.10), radii=(0.06, 0.04), duration=PRINT_TIME):
    s = np.clip(t/duration, 0, 1)
    h = 10*s**3 - 15*s**4 + 6*s**5
    dh = (30*s**2 - 60*s**3 + 30*s**4) / duration
    ddh = (60*s - 180*s**2 + 120*s**3) / duration**2

    th, w, b = 2*np.pi*h, 2*np.pi*dh, 2*np.pi*ddh
    radial = np.array([np.cos(th), np.sin(th)])
    tangent = np.array([-np.sin(th), np.cos(th)])
    radii = np.array(radii)

    p = np.array(center) + radii*radial
    v = radii*tangent*w
    acc = radii*(tangent*b - radial*w**2)
    return p, v, acc