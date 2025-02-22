import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Hodgkin-Huxley parameters
Cm = 1  # membrane capacitance, uF/cm^2

# ENa = 55.0  # sodium reversal potential, mV
# EK = -77.0  # potassium reversal potential, mV
# EL = -65.4  # leak reversal potential, mV

# gNa = 40.0  # sodium conductance, mS/cm^2
# gK = 35.0  # potassium conductance, mS/cm^2
# gL = 0.3  # leak conductance, mS/cm^2

# parameters from
#  https://webpages.uidaho.edu/rwells/techdocs/Biological%20Signal%20Processing/Chapter%2003%20The%20Hodgkin-Huxley%20Model.pdf
#  check original paper
ENa, EK, EL = 115, -12, 10.613
gNa, gK, gL = 120, 36, 0.3
threshold = 7.24
th_factor = 1.0


# Input current: Heaviside step current
def I_inj(t):
    global threshold, th_factor
    return threshold * th_factor if t > 10 and t < 4000 else 0.0


# Gating variable dynamics
def alpha_m(V):
    return 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1)


def beta_m(V):
    return 4.0 * np.exp(-V / 18)


def alpha_h(V):
    return 0.07 * np.exp(-V / 20)


def beta_h(V):
    return 1.0 / (np.exp((30 - V) / 10) + 1)


def alpha_n(V):
    return 0.01 * (10 - V) / (np.exp((10 - V) / 10) - 1)


def beta_n(V):
    return 0.125 * np.exp(-V / 80)


# Hodgkin-Huxley equations
def hh_model(t, y):
    V, m, h, n = y

    # Compute ionic currents
    INa = gNa * m**3 * h * (V - ENa)
    IK = gK * n**4 * (V - EK)
    IL = gL * (V - EL)

    # Compute gating variable derivatives
    dVdt = (I_inj(t) - INa - IK - IL) / Cm
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n

    return [dVdt, dmdt, dhdt, dndt]


def get_frequency(V):
    theta = 10
    crossings = np.sum((V[:-1] <= theta) & (V[1:] > theta))
    return (crossings - 1) / 500


# Initial conditions
# V0 = -65.0
# m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
# h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
# n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
V0, m0, h0, n0 = 0, 0, 0, 0

# Time span for simulation
t_span = (0, 300)
t_eval = np.linspace(*t_span, 10000)

# Solve the system
fa = []
f = []
for i in range(100):
    th_factor = 0.1 * i
    sol = solve_ivp(hh_model, t_span, [V0, m0, h0, n0], t_eval=t_eval, method="RK45")
    fa.append(th_factor)
    f.append(get_frequency(sol.y[0]))


plt.figure(figsize=(5, 5))
plt.plot(fa, f, "o")
plt.xlabel("Input current density (uA cm^-2)")
plt.ylabel("Frequency (kHz)")
plt.savefig("frequency.pdf", bbox_inches="tight", transparent=True, pad_inches=0)

th_factor = 0.5
sol = solve_ivp(hh_model, t_span, [V0, m0, h0, n0], t_eval=t_eval, method="RK45")
plt.figure(figsize=(5, 5))
plt.plot(sol.t, sol.y[0])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.savefig("hh_before.pdf", bbox_inches="tight", transparent=True, pad_inches=0)

th_factor = 1.1
sol = solve_ivp(hh_model, t_span, [V0, m0, h0, n0], t_eval=t_eval, method="RK45")
plt.figure(figsize=(5, 5))
plt.plot(sol.t, sol.y[0])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.savefig("hh_after.pdf", bbox_inches="tight", transparent=True, pad_inches=0)

# plt.show()
