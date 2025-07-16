"""Hodking Huxley's neuron model."""

# ==============================================================================
# IMPORTS
# ==============================================================================
import pathlib
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ==============================================================================
# Hodgkin-Huxley's neuron model
# ==============================================================================
# ------------------------------------------------------------------------------
# Parameters:
#  https://webpages.uidaho.edu/rwells/techdocs/Biological%20Signal%20Processing/Chapter%2003%20The%20Hodgkin-Huxley%20Model.pdf
#  check original paper
# ------------------------------------------------------------------------------
Cm: int = 1.0  # membrane capacitance, uF/cm^2

ENa, EK, EL = 115.0, -12.0, 10.613
gNa, gK, gL = 120.0, 36.0, 0.3  # noqa: N816
threshold = 7.24
th_factor = 1.0


# ------------------------------------------------------------------------------
# ODE functions:
# ------------------------------------------------------------------------------
# Input current: Heaviside step current
def I_inj(t: float, threshold: float, th_factor: float) -> float:  # noqa: N802
    """_summary_.

    Args:
        t (float): _description_
        threshold (float): _description_
        th_factor (float): _description_

    Returns:
        float: _description_

    """
    return (threshold * th_factor) if t > 10.0 and t < 4000.0 else 0.0  # noqa: PLR2004


# Gating variable dynamics
def alpha_m(V: float) -> float:  # noqa: N803
    """_summary_.

    Args:
        V (float): _description_

    Returns:
        float: _description_

    """
    return 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1)


def beta_m(V: float) -> float:  # noqa: N803
    """_summary_.

    Args:
        V (float): _description_

    Returns:
        float: _description_

    """
    return 4.0 * np.exp(-V / 18)


def alpha_h(V: float) -> float:  # noqa: N803
    """_summary_.

    Args:
        V (float): _description_

    Returns:
        float: _description_

    """
    return 0.07 * np.exp(-V / 20)


def beta_h(V: float) -> float:  # noqa: N803
    """_summary_.

    Args:
        V (float): _description_

    Returns:
        float: _description_

    """
    return 1.0 / (np.exp((30 - V) / 10) + 1)


def alpha_n(V: float) -> float:  # noqa: N803
    """_summary_.

    Args:
        V (float): _description_

    Returns:
        float: _description_

    """
    return 0.01 * (10 - V) / (np.exp((10 - V) / 10) - 1)


def beta_n(V: float) -> float:  # noqa: N803
    """_summary_.

    Args:
        V (float): _description_

    Returns:
        float: _description_

    """
    return 0.125 * np.exp(-V / 80)


# ------------------------------------------------------------------------------
# Hodgkin-Huxley ODE system
# ------------------------------------------------------------------------------
def hh_model(
    t: float,
    y: list[float, float, float, float],
    threshold: float,
    th_factor: float,
) -> list[float, float, float, float]:
    """_summary_.

    Args:
        t (float): _description_
        y (list[float, float, float, float]): _description_
        threshold (float): _description_
        th_factor (float): _description_

    Returns:
        list[float, float, float, float]: _description_

    """
    V, m, h, n = y  # noqa: N806

    # Compute ionic currents
    INa: float = gNa * m**3 * h * (V - ENa)  # noqa: N806
    IK: float = gK * n**4 * (V - EK)  # noqa: N806
    IL: float = gL * (V - EL)  # noqa: N806

    # Compute gating variable derivatives
    dVdt: float = (I_inj(t, threshold, th_factor) - INa - IK - IL) / Cm  # noqa: N806
    dmdt: float = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt: float = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt: float = alpha_n(V) * (1 - n) - beta_n(V) * n

    return [dVdt, dmdt, dhdt, dndt]


def get_frequency(V: float) -> float:  # noqa: N803
    """_summary_.

    Args:
        V (float): _description_

    Returns:
        float: _description_

    """
    theta = 10
    crossings = np.sum((V[:-1] <= theta) & (V[1:] > theta))
    return (crossings - 1) / 500


# ==============================================================================
# Paper Figure 1
# ==============================================================================
def main(outdir: str) -> None:
    """_summary_.

    Args:
        outdir (str): _description_

    """
    # --------------------------------------------------------------------------
    # Simulation settings
    # --------------------------------------------------------------------------
    # Time span for simulation
    t_span = (0, 200)
    t_eval = np.linspace(*t_span, 10000)

    # --------------------------------------------------------------------------
    # Figure 1c
    # --------------------------------------------------------------------------
    # ~ Init voltage -100 (orange)
    # Initial conditions
    V0 = -100  # noqa: N806
    m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
    h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
    n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))

    # Solve the system
    fa_100 = []
    f_100 = []
    for i in range(100):
        th_factor = 0.1 * i
        sol = solve_ivp(
            hh_model,
            t_span,
            [V0, m0, h0, n0],
            t_eval=t_eval,
            method="RK45",
            args=(threshold, th_factor),
        )
        fa_100.append(th_factor)
        f_100.append(get_frequency(sol.y[0]))

    # --------------------------------------------------------------------------
    # ~ Init voltage 0 (blue)
    # Initial conditions
    V0, m0, h0, n0 = 0, 0, 0, 0  # noqa: N806

    # Solve the system
    fa_0 = []
    f_0 = []
    for i in range(100):
        th_factor = 0.1 * i
        sol = solve_ivp(
            hh_model,
            t_span,
            [V0, m0, h0, n0],
            t_eval=t_eval,
            method="RK45",
            args=(threshold, th_factor),
        )
        fa_0.append(th_factor)
        f_0.append(get_frequency(sol.y[0]))

    # --------------------------------------------------------------------------
    # ~ Export Figure
    plt.figure(figsize=(6, 5))
    plt.plot(fa_0, f_0, "o", label="Initial voltage 0mV", markersize=6)
    plt.plot(
        fa_100,
        f_100,
        "o",
        label="Initial voltage -100mV",
        markersize=6,
        alpha=0.7,
    )
    plt.xlabel("Input current density (uA/cm²)", fontsize=16)
    plt.ylabel("Frequency (kHz)", fontsize=16)
    plt.legend(prop={"size": 16})
    plt.savefig(
        f"{outdir}/fig1c-frequency.pdf",
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
    )

    # --------------------------------------------------------------------------
    # Figure 1a
    # --------------------------------------------------------------------------
    # Initial conditions
    V0, m0, h0, n0 = 0, 0, 0, 0  # noqa: N806
    th_factor = 0.5
    sol = solve_ivp(
        hh_model,
        t_span,
        [V0, m0, h0, n0],
        t_eval=t_eval,
        method="RK45",
        args=(threshold, th_factor),
    )
    plt.figure(figsize=(4, 5))
    plt.plot(sol.t, sol.y[0])
    plt.xlabel("Time (ms)", fontsize=16)
    plt.ylabel("Voltage (mV)", fontsize=16)
    plt.savefig(
        f"{outdir}/fig1a-below.pdf",
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
    )

    # --------------------------------------------------------------------------
    # Figure 1b
    # --------------------------------------------------------------------------
    th_factor = 1.1
    sol = solve_ivp(
        hh_model,
        t_span,
        [V0, m0, h0, n0],
        t_eval=t_eval,
        method="RK45",
        args=(threshold, th_factor),
    )
    plt.figure(figsize=(4, 5))
    plt.plot(sol.t, sol.y[0])
    plt.xlabel("Time (ms)", fontsize=16)
    plt.ylabel("Voltage (mV)", fontsize=16)
    plt.savefig(
        f"{outdir}/fig1b-above.pdf",
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
    )


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main(
        pathlib.Path(__file__).parent.resolve() / f"../{sys.argv[1]}",
    )
