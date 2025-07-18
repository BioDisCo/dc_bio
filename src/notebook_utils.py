"""Code for the Jupyter notebook for the interactive cells.

This code is not mean to be read by another human-being.
For details on the implementations, please refer to other `src/*.py` files.
"""

# ==============================================================================
# Import
# ==============================================================================
import random
import threading
import time
from typing import Any, Callable, Literal

import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import HTML, clear_output, display

from src.chemical_reaction_network import (
    simulate_abc,
    simulate_mutual_annihilation,
)
from src.consensus import (
    consensus,
    generate_graphs,
    generate_graphs_sequence,
    graph_approachextreme,
    graph_mean,
    graph_midextremes,
    graph_midpoint,
)
from src.differential_evolution import (
    custom_strategy_fn,
    differential_evolution,
)
from src.gradient_descent import (
    gradient_descent,
    gradient_descent_momentum,
    gradient_descent_noise,
)
from src.hodgkin_huxley import (
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
    hh_model,
    solve_ivp,
    threshold,
)
from src.maximal_independent_set import (
    graph_step,
    maximal_independent_set,
)
from src.particle_swarm import (
    particle_swarm_optimization,
)


# ==============================================================================
# Hodgkin-Huxley Neuron Model
# ==============================================================================
def interact_hodgkin_huxley(duration: float = 200.0, steps: int = 500):
    clear_output()

    # --- Default values ---
    defaults = {
        "thf": 0.5,
        "v0": 0.0,
        "m0": 0.0,
        "h0": 0.0,
        "n0": 0.0,
    }

    # --- Widgets ---
    th_field = widgets.FloatText(
        value=threshold,
        description="Threshold",
    )
    v0_slider = widgets.FloatSlider(
        value=defaults["v0"],
        min=-100,
        max=100,
        step=1.0,
        description="V0",
    )
    thf_slider = widgets.FloatSlider(
        value=defaults["thf"],
        min=-1,
        max=1,
        step=0.01,
        description="Factor",
    )
    m0_slider = widgets.FloatSlider(
        value=defaults["m0"],
        min=-1,
        max=1,
        step=0.01,
        description="m0",
    )
    h0_slider = widgets.FloatSlider(
        value=defaults["h0"],
        min=-1,
        max=1,
        step=0.01,
        description="h0",
    )
    n0_slider = widgets.FloatSlider(
        value=defaults["n0"],
        min=-1,
        max=1,
        step=0.01,
        description="n0",
    )

    reset_button = widgets.Button(description="Reset")
    auto_checkbox = widgets.Checkbox(value=False, description="Auto")

    # --- Layout ---
    sliders_row1 = widgets.HBox([th_field, thf_slider])
    sliders_row2 = widgets.HBox(
        [m0_slider, h0_slider, n0_slider, auto_checkbox],
    )
    button_row = widgets.HBox([reset_button])
    ui = widgets.VBox(
        [sliders_row1, v0_slider, sliders_row2, button_row],
    )
    display(ui)

    # --- Debounce timer ---
    debounce_timer = None
    debounce_delay = 0.5  # seconds

    # --- Run simulation and plot ---
    def run_sim() -> None:
        nonlocal debounce_timer
        t_span = (0, duration)
        t_eval = np.linspace(*t_span, steps)

        v0 = v0_slider.value
        threshold = th_field.value
        th_factor = thf_slider.value

        if auto_checkbox.value:
            m0 = alpha_m(v0) / (alpha_m(v0) + beta_m(v0))
            h0 = alpha_h(v0) / (alpha_h(v0) + beta_h(v0))
            n0 = alpha_n(v0) / (alpha_n(v0) + beta_n(v0))
            m0_slider.value = m0
            h0_slider.value = h0
            n0_slider.value = n0
        else:
            m0 = m0_slider.value
            h0 = h0_slider.value
            n0 = n0_slider.value

        sol = solve_ivp(
            hh_model,
            t_span,
            [v0, m0, h0, n0],
            t_eval=t_eval,
            method="RK45",
            args=(threshold, th_factor),
        )

        # --- Plot the results ---
        clear_output(wait=True)
        display(ui)

        plt.figure(figsize=(6, 4))
        plt.plot(sol.t, sol.y[0])
        y_vals = sol.y[0]
        y_min, y_max = np.min(y_vals), np.max(y_vals)
        pad = 0.1 * (y_max - y_min) if y_max != y_min else 1.0
        plt.ylim(y_min - pad, y_max + pad)

        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.grid()
        plt.tight_layout()
        plt.show()

    # --- Debounced update ---
    def on_slider_change(_: Any) -> None:  # noqa: ANN401
        nonlocal debounce_timer
        if debounce_timer is not None:
            debounce_timer.cancel()
        debounce_timer = threading.Timer(debounce_delay, run_sim)
        debounce_timer.start()

    def on_text_change(_: Any) -> None:  # noqa: ANN401
        run_sim()

    def on_auto_toggle(_: Any) -> None:  # noqa: ANN401
        auto = auto_checkbox.value
        for w in [m0_slider, h0_slider, n0_slider]:
            w.disabled = auto
        run_sim()

    def on_reset_click(_: Any) -> None:  # noqa: ANN401
        th_field.value = threshold
        v0_slider.value = defaults["v0"]
        thf_slider.value = defaults["thf"]
        m0_slider.value = defaults["m0"]
        h0_slider.value = defaults["h0"]
        n0_slider.value = defaults["n0"]
        auto_checkbox.value = False
        run_sim()

    # --- Register events ---
    for slider in [
        v0_slider,
        m0_slider,
        h0_slider,
        n0_slider,
        thf_slider,
    ]:
        slider.observe(on_slider_change, names="value")
    th_field.observe(on_text_change)
    reset_button.on_click(on_reset_click)
    auto_checkbox.observe(on_auto_toggle, names="value")

    # --- Initial plot ---
    run_sim()


# ==============================================================================
# Gradient Descent
# ==============================================================================
def interact_gradient_descent(
    f: Callable[[float, float], float],
    df_dx: Callable[[float, float], float],
    df_dy: Callable[[float, float], float],
) -> None:
    clear_output()

    # --------------------------------------------------------------------------
    # Widgets
    # --------------------------------------------------------------------------
    iteration_field: widgets.IntText = widgets.IntText(
        value=200,
        description="Iterations",
    )
    learning_rate_slide: widgets.FloatLogSlider = widgets.FloatSlider(
        value=0.05,
        min=0,
        max=1,
        step=1e-3,
        description="Learning rate",
    )
    init_x0_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=1.5,
        min=-2,
        max=2,
        step=0.01,
        description="x0",
    )
    init_y0_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=1.5,
        min=-2,
        max=2,
        step=0.01,
        description="y0",
    )
    noise_slider: widgets.IntSlider = widgets.IntSlider(
        value=2,
        min=0,
        max=10,
        step=1,
        description="Noise",
    )
    momentum_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=0.8,
        min=0,
        max=1,
        step=0.01,
        description="Momentum",
    )
    seed_field: widgets.IntText = widgets.IntText(
        value=42,
        min=0,
        max=100,
        step=1,
        description="Seed",
    )
    run_button: widgets.Button = widgets.Button(description="Run")
    animation_checkbox: widgets.Checkbox = widgets.Checkbox(
        value=False,
        description="Animation",
    )

    # --------------------------------------------------------------------------
    # GUI
    # --------------------------------------------------------------------------
    widget_row0 = widgets.HBox([iteration_field, learning_rate_slide])
    widget_row1 = widgets.HBox([init_x0_slider, init_y0_slider])
    widget_row2 = widgets.HBox(
        [noise_slider, momentum_slider, seed_field],
    )
    widget_row3 = widgets.HBox(
        [run_button, animation_checkbox],
    )

    plot_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="400px"),
    )
    ui = widgets.VBox(
        [widget_row0, widget_row1, widget_row2, widget_row3, plot_frame],
    )
    display(ui)

    # Shared mutable state
    state: dict[str, bool] = {"cancel": False, "animate": True}

    def animation():
        state["animate"] = not state["animate"]

    # --------------------------------------------------------------------------
    # Pre-compute gradient maps
    # --------------------------------------------------------------------------
    x_vals: np.typing.NDArray = np.linspace(-2, 2, 100)
    y_vals: np.typing.NDArray = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_vals, y_vals)  # noqa: N806
    Z: np.typing.NDArray = f(X, Y)  # noqa: N806

    min_index = np.unravel_index(np.argmin(Z), Z.shape)
    min_x = X[min_index]
    min_y = Y[min_index]

    # --------------------------------------------------------------------------
    # Init plot
    # --------------------------------------------------------------------------
    titles: list[str] = ["Standard", "with momentum", "with noisy"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    with plot_frame:
        plt.ion()
        lines = []
        for ax, title in zip(axes, titles):
            # Plot background
            contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("x", fontsize=14)
            ax.set_ylabel("y", fontsize=14)
            ax.tick_params(labelsize=12)

            ax.plot([min_x], [min_y], color="orange", marker="o", zorder=5)

            # Init path line
            (line,) = ax.plot(
                [],
                [],
                color="red",
                marker="o",
                linestyle="dashed",
                label="Descent path",
            )
            lines.append(line)

        # Add shared colorbar
        cbar = fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_label("f(x, y)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        display(fig)

    # --------------------------------------------------------------------------
    # Run function
    # --------------------------------------------------------------------------
    def run() -> None:
        run_button.disabled = True
        animation_checkbox.disabled = True
        for line in lines:
            line.set_data([], [])

        # ~ Compute gradient descents
        history_gd: list[tuple[float, float]] = gradient_descent(
            df_dx,
            df_dy,
            init_x0_slider.value,
            init_y0_slider.value,
            learning_rate=learning_rate_slide.value,
            max_iter=iteration_field.value,
        )
        history_gdn: list[tuple[float, float]] = gradient_descent_noise(
            df_dx,
            df_dy,
            init_x0_slider.value,
            init_y0_slider.value,
            learning_rate=learning_rate_slide.value,
            max_iter=iteration_field.value,
            seed=seed_field.value,
            noise_step=noise_slider.value,
        )
        history_gdm: list[tuple[float, float]] = gradient_descent_momentum(
            df_dx,
            df_dy,
            init_x0_slider.value,
            init_y0_slider.value,
            learning_rate=learning_rate_slide.value,
            max_iter=iteration_field.value,
            momentum=momentum_slider.value,
        )

        # ~ Plot the results
        histories: list[list[tuple[float, float]]] = [
            history_gd,
            history_gdn,
            history_gdm,
        ]

        # ------------------------------------------------------------------
        # Draw plot
        # ------------------------------------------------------------------
        if animation_checkbox.value:
            n_steps = max(len(h) for h in histories)
            for i in range(n_steps):
                for j, history in enumerate(histories):
                    history_x: list[float] = [
                        point[0] for point in history[: i + 1]
                    ]
                    history_y: list[float] = [
                        point[1] for point in history[: i + 1]
                    ]
                    lines[j].set_data(history_x, history_y)

                with plot_frame:
                    plot_frame.clear_output(wait=True)
                    display(fig)
        else:
            for j, history in enumerate(histories):
                history_x: list[float] = [point[0] for point in history]
                history_y: list[float] = [point[1] for point in history]
                lines[j].set_data(history_x, history_y)

            with plot_frame:
                plot_frame.clear_output(wait=True)
                display(fig)

        run_button.disabled = False
        animation_checkbox.disabled = False

    # --------------------------------------------------------------------------
    # Deploy observers
    # --------------------------------------------------------------------------
    run_button.on_click(lambda _: run())
    animation_checkbox.observe(lambda _: animation(), names="value")
    plt.close()


# ==============================================================================
# Differential Evolution
# ==============================================================================
def interact_differential_evolution(
    f: Callable[[float, float], float],
) -> None:
    clear_output()

    # --------------------------------------------------------------------------
    # Widgets
    # --------------------------------------------------------------------------
    iteration_field: widgets.IntText = widgets.IntText(
        value=50,
        description="Iterations",
    )
    population_slider: widgets.IntSlider = widgets.IntSlider(
        value=5,
        min=1,
        max=100,
        step=1,
        description="Population",
    )
    mutation_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=0.7,
        min=0,
        max=1,
        step=0.01,
        description="Mutation",
    )
    recombination_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=0.9,
        min=0,
        max=1,
        step=0.01,
        description="Recombination",
    )
    seed_field: widgets.IntText = widgets.IntText(
        value=35,
        min=0,
        max=100,
        step=1,
        description="Seed",
    )
    run_button: widgets.Button = widgets.Button(description="Run")
    animation_checkbox: widgets.Checkbox = widgets.Checkbox(
        value=False,
        description="Animation",
    )

    # --------------------------------------------------------------------------
    # GUI
    # --------------------------------------------------------------------------
    widget_row0 = widgets.HBox([iteration_field])
    widget_row1 = widgets.HBox(
        [population_slider, mutation_slider, recombination_slider, seed_field],
    )
    widget_row2 = widgets.HBox(
        [run_button, animation_checkbox],
    )

    plot_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="400px"),
    )
    ui = widgets.VBox(
        [widget_row0, widget_row1, widget_row2, plot_frame],
    )
    display(ui)

    # Shared mutable state
    state: dict[str, bool] = {"cancel": False, "animate": True}

    def animation():
        state["animate"] = not state["animate"]

    # --------------------------------------------------------------------------
    # Pre-compute gradient maps
    # --------------------------------------------------------------------------
    x_vals: np.typing.NDArray = np.linspace(-2, 2, 100)
    y_vals: np.typing.NDArray = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_vals, y_vals)  # noqa: N806
    Z: np.typing.NDArray = f(X, Y)  # noqa: N806

    min_index = np.unravel_index(np.argmin(Z), Z.shape)
    min_x = X[min_index]
    min_y = Y[min_index]

    # --------------------------------------------------------------------------
    # Init plot
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    with plot_frame:
        plt.ion()
        lines = []
        # Plot background
        contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.tick_params(labelsize=12)

        ax.plot([min_x], [min_y], color="orange", marker="o", zorder=5)

        # Init path line
        ax_lines = []
        for _ in range(population_slider.max):
            (line,) = ax.plot(
                [],
                [],
                # color="red",
                marker="o",
                linestyle="dashed",
                linewidth=1.5,
            )
            ax_lines.append(line)
        lines.append(ax_lines)

        # Add shared colorbar
        cbar = fig.colorbar(contour, ax=ax, shrink=0.95)
        cbar.set_label("f(x, y)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        display(fig)

    # --------------------------------------------------------------------------
    # Run function
    # --------------------------------------------------------------------------
    def run() -> None:
        run_button.disabled = True
        animation_checkbox.disabled = True
        for ax_lines in lines:
            for line in ax_lines:
                line.set_data([], [])

        # ~ Compute gradient descents
        history: list = differential_evolution(
            f,
            [(-2, 2), (-2, 2)],
            custom_strategy_fn,
            popsize=population_slider.value,
            mutation=mutation_slider.value,
            recombination=recombination_slider.value,
            seed=seed_field.value,
            max_iter=iteration_field.value,
            updates=None,
        )

        # ~ Plot the results
        histories: list[list[tuple[float, float]]] = [history]

        # ------------------------------------------------------------------
        # Draw plot
        # ------------------------------------------------------------------
        if animation_checkbox.value:
            n_steps = max(len(h) for h in histories)
            for i in range(n_steps):
                for j, history in enumerate(histories):
                    for pn in range(len(history[0][0])):
                        history_x: list[float] = [
                            point[0][pn] for point in history[: i + 1]
                        ]
                        history_y: list[float] = [
                            point[1][pn] for point in history[: i + 1]
                        ]
                        lines[j][pn].set_data(history_x, history_y)

                with plot_frame:
                    plot_frame.clear_output(wait=True)
                    display(fig)
                    time.sleep(0.1)
        else:
            for j, history in enumerate(histories):
                for pn in range(len(history[0][0])):
                    history_x: list[float] = [point[0][pn] for point in history]
                    history_y: list[float] = [point[1][pn] for point in history]
                    lines[j][pn].set_data(history_x, history_y)

            with plot_frame:
                plot_frame.clear_output(wait=True)
                display(fig)

        run_button.disabled = False
        animation_checkbox.disabled = False

    # --------------------------------------------------------------------------
    # Deploy observers
    # --------------------------------------------------------------------------
    run_button.on_click(lambda _: run())
    animation_checkbox.observe(lambda _: animation(), names="value")
    plt.close()


# ==============================================================================
# Particle Swarm
# ==============================================================================
def interact_particle_swarm(
    f: Callable[[float, float], float],
) -> None:
    clear_output()

    # --------------------------------------------------------------------------
    # Widgets
    # --------------------------------------------------------------------------
    iteration_field: widgets.IntText = widgets.IntText(
        value=50,
        description="Iterations",
    )
    population_slider: widgets.IntSlider = widgets.IntSlider(
        value=10,
        min=1,
        max=100,
        step=1,
        description="Population",
    )
    w_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=0.05,
        min=0,
        max=1,
        step=0.01,
        description="W",
    )
    c1_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=0.15,
        min=0,
        max=2,
        step=0.01,
        description="C1",
    )
    c2_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=0.15,
        min=0,
        max=2,
        step=0.01,
        description="C2",
    )
    seed_field: widgets.IntText = widgets.IntText(
        value=42,
        min=0,
        max=100,
        step=1,
        description="Seed",
    )
    run_button: widgets.Button = widgets.Button(description="Run")
    animation_checkbox: widgets.Checkbox = widgets.Checkbox(
        value=False,
        description="Animation",
    )

    # --------------------------------------------------------------------------
    # GUI
    # --------------------------------------------------------------------------
    widget_row0 = widgets.HBox([iteration_field])
    widget_row1 = widgets.HBox(
        [population_slider],
    )
    widget_row2 = widgets.HBox(
        [w_slider, c1_slider, c2_slider, seed_field],
    )
    widget_row3 = widgets.HBox(
        [run_button, animation_checkbox],
    )

    plot_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="400px"),
    )
    ui = widgets.VBox(
        [widget_row0, widget_row1, widget_row2, widget_row3, plot_frame],
    )
    display(ui)

    # Shared mutable state
    state: dict[str, bool] = {"cancel": False, "animate": True}

    def animation():
        state["animate"] = not state["animate"]

    # --------------------------------------------------------------------------
    # Pre-compute gradient maps
    # --------------------------------------------------------------------------
    x_vals: np.typing.NDArray = np.linspace(-2, 2, 100)
    y_vals: np.typing.NDArray = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_vals, y_vals)  # noqa: N806
    Z: np.typing.NDArray = f(X, Y)  # noqa: N806

    min_index = np.unravel_index(np.argmin(Z), Z.shape)
    min_x = X[min_index]
    min_y = Y[min_index]

    # --------------------------------------------------------------------------
    # Init plot
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    with plot_frame:
        plt.ion()
        lines = []
        # Plot background
        contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.tick_params(labelsize=12)

        ax.plot([min_x], [min_y], color="orange", marker="o", zorder=5)

        # Init path line
        ax_lines = []
        for _ in range(population_slider.max):
            (line,) = ax.plot(
                [],
                [],
                # color="red",
                marker="o",
                linestyle="dashed",
            )
            ax_lines.append(line)
        lines.append(ax_lines)

        # Add shared colorbar
        cbar = fig.colorbar(contour, ax=ax, shrink=0.95)
        cbar.set_label("f(x, y)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        display(fig)

    # --------------------------------------------------------------------------
    # Run function
    # --------------------------------------------------------------------------
    def run() -> None:
        run_button.disabled = True
        animation_checkbox.disabled = True
        for ax_lines in lines:
            for line in ax_lines:
                line.set_data([], [])

        # ~ Compute gradient descents
        history: list = particle_swarm_optimization(
            f,
            (-2, 2),
            w_slider.value,
            c1_slider.value,
            c2_slider.value,
            num_particles=population_slider.value,
            seed=seed_field.value,
            max_iter=iteration_field.value,
        )

        # ~ Plot the results
        histories: list[list[tuple[float, float]]] = [history]

        # ------------------------------------------------------------------
        # Draw plot
        # ------------------------------------------------------------------
        if animation_checkbox.value:
            n_steps = max(len(h) for h in histories)
            for i in range(n_steps):
                for j, history in enumerate(histories):
                    for pn in range(len(history[0])):
                        history_x = [
                            history[i][pn][0]
                            for i in range(len(history[: i + 1]))
                        ]
                        history_y = [
                            history[i][pn][1]
                            for i in range(len(history[: i + 1]))
                        ]
                        lines[j][pn].set_data(history_x, history_y)

                with plot_frame:
                    plot_frame.clear_output(wait=True)
                    display(fig)
                    time.sleep(0.1)
        else:
            for j, history in enumerate(histories):
                for pn in range(len(history[0])):
                    history_x = [history[i][pn][0] for i in range(len(history))]
                    history_y = [history[i][pn][1] for i in range(len(history))]
                    lines[j][pn].set_data(history_x, history_y)

            with plot_frame:
                plot_frame.clear_output(wait=True)
                display(fig)

        run_button.disabled = False
        animation_checkbox.disabled = False

    # --------------------------------------------------------------------------
    # Deploy observers
    # --------------------------------------------------------------------------
    run_button.on_click(lambda _: run())
    animation_checkbox.observe(lambda _: animation(), names="value")
    plt.close()


# ==============================================================================
# Chemical Reaction Network
# ==============================================================================
def interact_crn_abc(duration: float = 10.0):
    clear_output()

    # --- Default values ---
    defaults = {
        "a0": 10,
        "b0": 10,
        "k1": 0.1,
        "k2": 0.1,
        "repeats": 10,
        "seed": 42,
    }

    # --- Widgets ---
    a0_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=defaults["a0"],
        min=0,
        max=100,
        step=0.5,
        description="A0",
    )
    b0_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=defaults["b0"],
        min=0,
        max=100,
        step=0.5,
        description="B0",
    )
    k1_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=defaults["k1"],
        min=0,
        max=2,
        step=0.01,
        description="k1",
    )
    k2_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=defaults["k2"],
        min=0,
        max=2,
        step=0.01,
        description="k2",
    )
    seed_field: widgets.IntText = widgets.IntText(
        value=defaults["seed"],
        min=0,
        max=100,
        step=1,
        description="Seed",
    )
    repeat_slider: widgets.IntSlider = widgets.IntSlider(
        value=defaults["repeats"],
        min=0,
        max=100,
        step=1,
        description="Repeats",
    )

    reset_button = widgets.Button(description="Reset")

    # --- Layout ---
    sliders_row1 = widgets.HBox([a0_slider, b0_slider])
    sliders_row2 = widgets.HBox(
        [k1_slider, k2_slider],
    )
    sliders_row3 = widgets.HBox(
        [seed_field, repeat_slider],
    )
    button_row = widgets.HBox([reset_button])
    ui = widgets.VBox(
        [sliders_row1, sliders_row2, sliders_row3, button_row],
    )
    display(ui)

    # --- Debounce timer ---
    debounce_timer = None
    debounce_delay = 0.5  # seconds

    # --- Run simulation and plot ---
    def run_sim() -> None:
        nonlocal debounce_timer

        sim_deterministic = simulate_abc(
            (a0_slider.value, b0_slider.value),
            (k1_slider.value, k2_slider.value),
            duration,
            repeats=1,
            simulation_method="deterministic",
            seed=seed_field.value,
        )

        sim_stochastic = simulate_abc(
            (a0_slider.value, b0_slider.value),
            (k1_slider.value, k2_slider.value),
            duration,
            repeats=repeat_slider.value,
            simulation_method="stochastic",
            seed=seed_field.value,
        )

        # --- Plot the results ---
        clear_output(wait=True)
        display(ui)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for ax, sim, mode in zip(
            axes,
            [sim_deterministic, sim_stochastic],
            ["deterministic", "stochastic"],
        ):
            linewidths = (1.5, 1) if mode == "deterministic" else (2.5, 2)

            for i, res in enumerate(sim):
                ax.plot(
                    res["Time"],
                    res["A"],
                    label="A" if i == 0 else "_",
                    color="red",
                    alpha=1 if i == 0 else 0.3,
                    linewidth=linewidths[0] if i == 0 else linewidths[1],
                )
                ax.plot(
                    res["Time"],
                    res["B"],
                    label="B" if i == 0 else "_",
                    color="blue",
                    alpha=1 if i == 0 else 0.3,
                    linewidth=linewidths[0] if i == 0 else linewidths[1],
                )
                ax.plot(
                    res["Time"],
                    res["C"],
                    label="C" if i == 0 else "_",
                    color="black",
                    alpha=1 if i == 0 else 0.3,
                    linewidth=linewidths[0] if i == 0 else linewidths[1],
                )

            ax.set_title(mode.capitalize())
            ax.set_xlabel("time")
            ax.set_ylabel(
                "count/volume" if mode == "deterministic" else "count",
            )
            ax.legend(frameon=False, loc=(0.8, 0.7))

        plt.tight_layout()
        plt.show()

    # --- Debounced update ---
    def on_slider_change(_: Any) -> None:  # noqa: ANN401
        nonlocal debounce_timer
        if debounce_timer is not None:
            debounce_timer.cancel()
        debounce_timer = threading.Timer(debounce_delay, run_sim)
        debounce_timer.start()

    def on_text_change(_: Any) -> None:  # noqa: ANN401
        run_sim()

    def on_reset_click(_: Any) -> None:  # noqa: ANN401
        a0_slider.value = defaults["a0"]
        b0_slider.value = defaults["b0"]
        k1_slider.value = defaults["k1"]
        k2_slider.value = defaults["k2"]
        seed_field.value = defaults["seed"]
        repeat_slider.value = defaults["repeats"]
        run_sim()

    # --- Register events ---
    for slider in [
        a0_slider,
        b0_slider,
        k1_slider,
        k2_slider,
        repeat_slider,
    ]:
        slider.observe(on_slider_change, names="value")
    seed_field.observe(on_text_change)
    reset_button.on_click(on_reset_click)

    # --- Initial plot ---
    run_sim()


def interact_crn_mutual_annihilation(duration: float = 10.0):
    clear_output()

    # --- Default values ---
    defaults = {
        "a0": 12,
        "b0": 8,
        "r": 100,
        "k1": 0.1,
        "k2": 0.1,
        "k3": 0.01,
        "k4": 0.01,
        "repeats": 10,
        "seed": 42,
    }

    # --- Widgets ---
    a0_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=defaults["a0"],
        min=0,
        max=100,
        step=0.5,
        description="A0",
    )
    b0_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=defaults["b0"],
        min=0,
        max=100,
        step=0.5,
        description="B0",
    )
    r_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=defaults["r"],
        min=0,
        max=1000,
        step=1,
        description="R",
    )
    k1_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=defaults["k1"],
        min=0,
        max=2,
        step=0.01,
        description="k1",
    )
    k2_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=defaults["k2"],
        min=0,
        max=2,
        step=0.01,
        description="k2",
    )
    k3_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=defaults["k3"],
        min=0,
        max=2,
        step=0.01,
        description="k3",
    )
    k4_slider: widgets.FloatSlider = widgets.FloatSlider(
        value=defaults["k4"],
        min=0,
        max=2,
        step=0.01,
        description="k4",
    )
    seed_field: widgets.IntText = widgets.IntText(
        value=defaults["seed"],
        min=0,
        max=100,
        step=1,
        description="Seed",
    )
    repeat_slider: widgets.IntSlider = widgets.IntSlider(
        value=defaults["repeats"],
        min=0,
        max=100,
        step=1,
        description="Repeats",
    )

    reset_button = widgets.Button(description="Reset")

    # --- Layout ---
    sliders_row1 = widgets.HBox([a0_slider, b0_slider, r_slider])
    sliders_row2 = widgets.HBox(
        [k1_slider, k2_slider, k3_slider, k4_slider],
    )
    sliders_row3 = widgets.HBox(
        [seed_field, repeat_slider],
    )
    button_row = widgets.HBox([reset_button])
    ui = widgets.VBox(
        [sliders_row1, sliders_row2, sliders_row3, button_row],
    )
    display(ui)

    # --- Debounce timer ---
    debounce_timer = None
    debounce_delay = 0.5  # seconds

    # --- Run simulation and plot ---
    def run_sim() -> None:
        nonlocal debounce_timer

        sim_deterministic = simulate_mutual_annihilation(
            (a0_slider.value, b0_slider.value, r_slider.value),
            (
                k1_slider.value,
                k2_slider.value,
                k3_slider.value,
                k4_slider.value,
            ),
            duration,
            repeats=1,
            simulation_method="deterministic",
            seed=seed_field.value,
        )

        sim_stochastic = simulate_mutual_annihilation(
            (a0_slider.value, b0_slider.value, r_slider.value),
            (
                k1_slider.value,
                k2_slider.value,
                k3_slider.value,
                k4_slider.value,
            ),
            duration,
            repeats=repeat_slider.value,
            simulation_method="stochastic",
            seed=seed_field.value,
        )

        # --- Plot the results ---
        clear_output(wait=True)
        display(ui)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for ax, sim, mode in zip(
            axes,
            [sim_deterministic, sim_stochastic],
            ["deterministic", "stochastic"],
        ):
            for i, res in enumerate(sim):
                ax.plot(
                    res["Time"],
                    res["A"],
                    label="A" if i == 0 else "_",
                    color="red",
                    alpha=1 if i == 0 else 0.3,
                    linewidth=1.5 if i == 0 else 1,
                )
                ax.plot(
                    res["Time"],
                    res["B"],
                    label="B" if i == 0 else "_",
                    color="blue",
                    alpha=1 if i == 0 else 0.3,
                    linewidth=1.5 if i == 0 else 1,
                )

            ax.set_ylim([0, r_slider.value + 1])
            ax.set_title(mode.capitalize())
            ax.set_xlabel("time")
            ax.set_ylabel(
                "count/volume" if mode == "deterministic" else "count",
            )
            ax.legend(frameon=False, loc=(0.8, 0.7))

        plt.tight_layout()
        plt.show()

    # --- Debounced update ---
    def on_slider_change(_: Any) -> None:  # noqa: ANN401
        nonlocal debounce_timer
        if debounce_timer is not None:
            debounce_timer.cancel()
        debounce_timer = threading.Timer(debounce_delay, run_sim)
        debounce_timer.start()

    def on_text_change(_: Any) -> None:  # noqa: ANN401
        run_sim()

    def on_reset_click(_: Any) -> None:  # noqa: ANN401
        a0_slider.value = defaults["a0"]
        b0_slider.value = defaults["b0"]
        r_slider.value = defaults["r"]
        k1_slider.value = defaults["k1"]
        k2_slider.value = defaults["k2"]
        k3_slider.value = defaults["k3"]
        k4_slider.value = defaults["k4"]
        seed_field.value = defaults["seed"]
        repeat_slider.value = defaults["repeats"]
        run_sim()

    # --- Register events ---
    for slider in [
        a0_slider,
        b0_slider,
        k1_slider,
        k2_slider,
        repeat_slider,
    ]:
        slider.observe(on_slider_change, names="value")
    seed_field.observe(on_text_change)
    reset_button.on_click(on_reset_click)

    # --- Initial plot ---
    run_sim()


# ==============================================================================
# Maximal Independent Sets
# ==============================================================================
def interact_maximal_independent_set() -> None:
    clear_output()

    # --------------------------------------------------------------------------
    # Widgets
    # --------------------------------------------------------------------------
    width_slider: widgets.IntSlider = widgets.IntSlider(
        value=4,
        min=1,
        max=10,
        step=1,
        description="Width",
    )
    height_slider: widgets.IntSlider = widgets.IntSlider(
        value=4,
        min=1,
        max=10,
        step=1,
        description="Height",
    )
    seed_field: widgets.IntText = widgets.IntText(
        value=42,
        min=0,
        max=100,
        step=1,
        description="Seed",
    )
    run_button: widgets.Button = widgets.Button(description="Run")
    animation_checkbox: widgets.Checkbox = widgets.Checkbox(
        value=False,
        description="Animation",
    )

    # --------------------------------------------------------------------------
    # GUI
    # --------------------------------------------------------------------------
    widget_row0 = widgets.HBox([width_slider, height_slider])
    widget_row2 = widgets.HBox(
        [run_button, animation_checkbox],
    )

    plot_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="400px"),
    )
    ui = widgets.VBox(
        [widget_row0, seed_field, widget_row2, plot_frame],
    )
    display(ui)

    # Shared mutable state
    state: dict[str, bool] = {"cancel": False, "animate": True}

    def animation():
        state["animate"] = not state["animate"]

    # --------------------------------------------------------------------------
    # Run function
    # --------------------------------------------------------------------------
    def run() -> None:
        run_button.disabled = True
        animation_checkbox.disabled = True

        np.random.seed(seed_field.value)

        # ~ Init Graph
        G: nx.Graph = nx.hexagonal_lattice_graph(
            width_slider.value,
            height_slider.value,
            periodic=False,
            with_positions=True,
        )

        max_degree: int = max(dict(G.degree()).values())
        mis_results, history = maximal_independent_set(
            G,
            phases=int(
                np.log(max_degree),
            )
            + 1,  # See https://www.science.org/doi/pdf/10.1126/science.1193210
            rounds_per_phase=20,
            step_func=graph_step,
            neighbors_ub=max_degree,
            seed=seed_field.value,
        )

        # ------------------------------------------------------------------
        # Draw plot
        # ------------------------------------------------------------------
        with plot_frame:
            fig, ax = plt.subplots()
            ax.axis("off")

            pos = nx.get_node_attributes(G, "pos")

            if animation_checkbox.value:
                for mis_state in history:
                    clear_output(wait=True)
                    node_colors = []
                    for node in G.nodes():
                        if "Term-MIS" in mis_state[node]:
                            node_colors.append("lightblue")
                        elif "Term-nonMIS" in mis_state[node]:
                            node_colors.append("white")
                        else:
                            node_colors.append("gray")

                    # draw
                    pos = nx.get_node_attributes(G, "pos")
                    nx.draw_networkx_nodes(
                        G,
                        pos,
                        node_color=node_colors,
                        edgecolors="black",
                        node_size=500,
                    )
                    nx.draw_networkx_edges(G, pos, edge_color="black")
                    plt.axis("off")
                    display(fig)
                    time.sleep(1)
            else:
                clear_output(wait=True)
                node_colors = []
                for node in G.nodes():
                    if "Term-MIS" in mis_results[node]:
                        node_colors.append("lightblue")
                    elif "Term-nonMIS" in mis_results[node]:
                        node_colors.append("white")
                    else:
                        node_colors.append("gray")

                # draw
                pos = nx.get_node_attributes(G, "pos")
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    node_color=node_colors,
                    edgecolors="black",
                    node_size=500,
                )
                nx.draw_networkx_edges(G, pos, edge_color="black")
                plt.axis("off")
                display(fig)
            plt.close(fig)

        run_button.disabled = False
        animation_checkbox.disabled = False

    # --------------------------------------------------------------------------
    # Deploy observers
    # --------------------------------------------------------------------------
    run_button.on_click(lambda _: run())
    animation_checkbox.observe(lambda _: animation(), names="value")
    run()
    plt.close()


# ==============================================================================
# Consensus
# ==============================================================================


# ------------------------------------------------------------------------------
# Graph generation
# ------------------------------------------------------------------------------
def interact_consensus_graphs() -> tuple[list[nx.DiGraph], list[nx.DiGraph]]:
    clear_output()

    # --------------------------------------------------------------------------
    # Widgets
    # --------------------------------------------------------------------------
    nodes_slider: widgets.IntSlider = widgets.IntSlider(
        value=10,
        min=1,
        max=10,
        step=1,
        description="#Nodes",
    )
    graphs_slider: widgets.IntSlider = widgets.IntSlider(
        value=3,
        min=1,
        max=5,
        step=1,
        description="#Graphs",
    )
    sequences_slider: widgets.IntSlider = widgets.IntSlider(
        value=10,
        min=1,
        max=50,
        step=1,
        description="|Sequence|",
    )
    seed_field: widgets.IntText = widgets.IntText(
        value=8,
        min=0,
        max=100,
        step=1,
        description="Seed",
    )
    text_field: widgets.Text = widgets.Label("Sequences:")
    run_button: widgets.Button = widgets.Button(description="Generate")

    # --------------------------------------------------------------------------
    # GUI
    # --------------------------------------------------------------------------
    widget_row0 = widgets.HBox([graphs_slider, nodes_slider])
    widget_row1 = widgets.HBox(
        [sequences_slider],
    )
    widget_row2 = widgets.HBox(
        [run_button],
    )

    plot_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="50px"),
    )
    sequence_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="50px"),
    )
    ui = widgets.VBox(
        [
            widget_row0,
            widget_row1,
            seed_field,
            widget_row2,
            plot_frame,
            text_field,
            sequence_frame,
        ],
    )
    display(ui)

    # State ~ mutable
    graphs: list[nx.DiGraph] = []
    sequence: list[nx.DiGraph] = []

    # --------------------------------------------------------------------------
    # Run function
    # --------------------------------------------------------------------------
    def run() -> None:
        run_button.disabled = True

        nodes_slider.value = max(nodes_slider.value, graphs_slider.value)

        graphs.clear()
        sequence.clear()

        graphs.extend(
            generate_graphs(
                nodes_slider.value,
                graphs_slider.value,
                seed=seed_field.value,
            ),
        )

        sequence.extend(
            generate_graphs_sequence(
                graphs,
                sequences_slider.value,
                seed=seed_field.value,
            ),
        )

        # ------------------------------------------------------------------
        # Draw plot
        # ------------------------------------------------------------------
        with plot_frame:
            clear_output(wait=True)
            fig, axes = plt.subplots(
                1,
                graphs_slider.max,
                figsize=(3 * graphs_slider.max, 3),
            )
            pos = nx.spring_layout(graphs[0], k=2, seed=seed_field.value)

            axes = axes.flatten()
            for ax in axes:
                ax.axis("off")

            for i, Gi in enumerate(graphs):
                nx.draw(
                    Gi,
                    pos,
                    with_labels=True,
                    node_color="lightblue",
                    edge_color="gray",
                    node_size=250,
                    font_size=10,
                    arrows=True,
                    arrowstyle="->",
                    arrowsize=7,
                    ax=axes[i],
                )
                axes[i].set_title(f"Graph {i + 1}")
            display(fig)
            plt.close(fig)

        with sequence_frame:
            clear_output(wait=True)

            fig, axes = plt.subplots(
                (1 + len(sequence) // 5) - (1 if len(sequence) % 5 == 0 else 0),
                5,
                figsize=(
                    3 * 5,
                    3
                    * (
                        1
                        + len(sequence) // 5
                        - (1 if len(sequence) % 5 == 0 else 0)
                    ),
                ),
            )

            axes = axes.flatten()
            for ax in axes:
                ax.axis("off")

            for i, Gi in enumerate(sequence):
                nx.draw(
                    Gi,
                    pos,
                    with_labels=True,
                    node_color="lightblue",
                    edge_color="gray",
                    node_size=250,
                    font_size=10,
                    arrows=True,
                    arrowstyle="->",
                    arrowsize=7,
                    ax=axes[i],
                )
                axes[i].set_title(f"Round - {i + 1}")
            display(fig)
            plt.close(fig)

        run_button.disabled = False

    # --------------------------------------------------------------------------
    # Deploy observers
    # --------------------------------------------------------------------------
    run_button.on_click(lambda _: run())

    run()

    return graphs, sequence


def interact_consensus_1D(
    sequence: list[nx.DiGraph], method: Literal["mean", "midpoint"],
) -> None:
    clear_output()

    # --------------------------------------------------------------------------
    # Widgets
    # --------------------------------------------------------------------------
    propagate_slider: widgets.IntSlider = widgets.IntSlider(
        value=2,
        min=2,
        max=10,
        step=1,
        description="Flooding steps",
    )
    seed_field: widgets.IntText = widgets.IntText(
        value=8,
        min=0,
        max=100,
        step=1,
        description="Seed",
    )
    run_button: widgets.Button = widgets.Button(description="Generate")

    # --------------------------------------------------------------------------
    # GUI
    # --------------------------------------------------------------------------
    plot1_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="50px"),
    )
    table1_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="50px"),
    )
    plot2_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="50px"),
    )
    table2_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="50px"),
    )
    widget_row0 = widgets.HBox([propagate_slider, seed_field, run_button])
    widget_row1 = widgets.HBox([plot1_frame, table1_frame])
    widget_row2 = widgets.HBox([plot2_frame, table2_frame])
    ui = widgets.VBox(
        [
            widget_row0,
            widget_row1,
            widget_row2,
        ],
    )
    display(ui)

    # --------------------------------------------------------------------------
    # Utils
    # --------------------------------------------------------------------------
    def render_table(history):
        print(history)
        html = "<table style='border-collapse: collapse;'>"
        html += (
            "<thead><tr>"
            + "".join(
                f"<th style='border:1px solid black;padding:4px'>{h if h != 0 else 'Rounds'}</th>"  # noqa: E501
                for h in range(len(history))
            )
            + "</tr></thead><tbody>"
        )
        for row in range(len(history[0])):
            html += (
                f"<tr><td style='border:1px solid black;padding:4px'>Node {row}</td>"
                + "".join(
                    f"<td style='border:1px solid black;padding:4px'>{cell:.4f}</td>"  # noqa: E501
                    for cell in history[0][row]
                )
                + "</tr>"
            )
        html += "</tbody></table>"
        return html

    def render_table(data: list[list[float]]) -> str:
        nb_rounds = len(data)
        nb_nodes = len(data[0]) if nb_rounds > 0 else 0

        # Transpose: rows → nodes, columns → rounds
        table = "<table style='border-collapse: collapse; border: 1px solid black;'>"

        # Header
        table += "<thead><tr><th style='border: 1px solid black; padding: 4px;'>Node \\ Round</th>"
        for r in range(nb_rounds):
            table += f"<th style='border: 1px solid black; padding: 4px;'>Round {r}</th>"
        table += "</tr></thead>"

        # Body: one row per node
        table += "<tbody>"
        for node in range(nb_nodes):
            table += f"<tr><td style='border: 1px solid black; padding: 4px;'>Node {node}</td>"
            for r in range(nb_rounds):
                val = data[r][node]
                table += f"<td style='border: 1px solid black; padding: 4px;'>{val:.3f}</td>"
            table += "</tr>"
        table += "</tbody></table>"

        return table

    # --------------------------------------------------------------------------
    # Run function
    # --------------------------------------------------------------------------
    def run() -> None:
        run_button.disabled = True

        random.seed(seed_field.value * 11)
        init_values_1D: dict[str, set[float]] = {
            node: {random.uniform(0, 1)} for node in sequence[0].nodes()
        }

        if method == "mean":
            history: list[list[float]] = consensus(
                graph_mean,
                sequence,
                init_values_1D,
                propagate_frequency=None,
            )

            history_propagate: list[list[float]] = consensus(
                graph_mean,
                sequence,
                init_values_1D,
                propagate_frequency=propagate_slider.value,
            )
        elif method == "midpoint":
            history: list[list[float]] = consensus(
                graph_midpoint,
                sequence,
                init_values_1D,
                propagate_frequency=None,
            )

            history_propagate: list[list[float]] = consensus(
                graph_midpoint,
                sequence,
                init_values_1D,
                propagate_frequency=propagate_slider.value,
            )
        else:
            msg = (
                f"Unknown method: {method}."
                "Please choose among `mean` or `midpoint`."
            )
            raise ValueError(msg)

        # ----------------------------------------------------------------------
        # Plot results
        # ----------------------------------------------------------------------
        with plot1_frame:
            clear_output(wait=True)
            plt.figure(figsize=(8, 3))
            plt.plot(
                range(len(history)),
                history,
                "-",
                color="gray",
                marker="o",
                alpha=0.6,
            )
            plt.grid()
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel("Round", fontsize=16)
            plt.ylabel("Value", fontsize=16)
            plt.show()

        with plot2_frame:
            clear_output(wait=True)
            plt.figure(figsize=(8, 3))
            plt.plot(
                range(len(history_propagate)),
                history_propagate,
                "-",
                color="gray",
                marker="o",
                alpha=0.6,
            )
            plt.grid()
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel("Round", fontsize=16)
            plt.ylabel("Value", fontsize=16)
            plt.show()

        # ----------------------------------------------------------------------
        # Draw tables
        # ----------------------------------------------------------------------
        with table1_frame:
            clear_output(wait=True)
            display(HTML(render_table(history)))

        with table2_frame:
            clear_output(wait=True)
            display(HTML(render_table(history_propagate)))

        run_button.disabled = False

    # --------------------------------------------------------------------------
    # Deploy observers
    # --------------------------------------------------------------------------
    run_button.on_click(lambda _: run())

    run()


def interact_consensus_2D(
    sequence: list[nx.DiGraph], method: Literal["midextreme", "approachextreme"],
) -> None:
    clear_output()

    # --------------------------------------------------------------------------
    # Widgets
    # --------------------------------------------------------------------------
    propagate_slider: widgets.IntSlider = widgets.IntSlider(
        value=2,
        min=2,
        max=10,
        step=1,
        description="Flooding steps",
    )
    seed_field: widgets.IntText = widgets.IntText(
        value=8,
        min=0,
        max=100,
        step=1,
        description="Seed",
    )
    run_button: widgets.Button = widgets.Button(description="Generate")

    # --------------------------------------------------------------------------
    # GUI
    # --------------------------------------------------------------------------
    plot1_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="50px"),
    )
    table1_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="50px"),
    )
    plot2_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="50px"),
    )
    table2_frame: widgets.Output = widgets.Output(
        layout=widgets.Layout(min_height="50px"),
    )
    widget_row0 = widgets.HBox([propagate_slider, seed_field, run_button])
    widget_row1 = widgets.HBox([plot1_frame, table1_frame])
    widget_row2 = widgets.HBox([plot2_frame, table2_frame])
    ui = widgets.VBox(
        [
            widget_row0,
            widget_row1,
            widget_row2,
        ],
    )
    display(ui)

    # --------------------------------------------------------------------------
    # Utils
    # --------------------------------------------------------------------------
    def render_table(history):
        print(history)
        html = "<table style='border-collapse: collapse;'>"
        html += (
            "<thead><tr>"
            + "".join(
                f"<th style='border:1px solid black;padding:4px'>{h if h != 0 else 'Rounds'}</th>"  # noqa: E501
                for h in range(len(history))
            )
            + "</tr></thead><tbody>"
        )
        for row in range(len(history[0])):
            html += (
                f"<tr><td style='border:1px solid black;padding:4px'>Node {row}</td>"  # noqa: E501
                + "".join(
                    f"<td style='border:1px solid black;padding:4px'>{cell:.4f}</td>"  # noqa: E501
                    for cell in history[0][row]
                )
                + "</tr>"
            )
        html += "</tbody></table>"
        return html

    def render_table(data: list[list[float]]) -> str:
        nb_rounds = len(data)
        nb_nodes = len(data[0]) if nb_rounds > 0 else 0

        # Transpose: rows → nodes, columns → rounds
        table = "<table style='border-collapse: collapse; border: 1px solid black;'>"  # noqa: E501

        # Header
        table += "<thead><tr><th style='border: 1px solid black; padding: 4px;'>Node \\ Round</th>"  # noqa: E501
        for r in range(nb_rounds):
            table += f"<th style='border: 1px solid black; padding: 4px;'>Round {r}</th>"  # noqa: E501
        table += "</tr></thead>"

        # Body: one row per node
        table += "<tbody>"
        for node in range(nb_nodes):
            table += f"<tr><td style='border: 1px solid black; padding: 4px;'>Node {node}</td>"  # noqa: E501
            for r in range(nb_rounds):
                val = data[r][node]
                table += f"<td style='border: 1px solid black; padding: 4px;'>({val[0]:.3f}, {val[1]:.3f})</td>"  # noqa: E501
            table += "</tr>"
        table += "</tbody></table>"

        return table

    # --------------------------------------------------------------------------
    # Run function
    # --------------------------------------------------------------------------
    def run() -> None:
        run_button.disabled = True

        random.seed(seed_field.value * 42)
        # ~ 2D
        init_values_2D: dict[str, set[tuple[float, float]]] = {
            node: {(random.uniform(0, 1), random.uniform(0, 1))}
            for node in sequence[0].nodes()
        }

        if method == "midextreme":
            history: list[list[tuple[float, float]]] = consensus(
                graph_midextremes,
                sequence,
                init_values_2D,
                propagate_frequency=None,
            )

            history_propagate: list[list[tuple[float, float]]] = consensus(
                graph_midextremes,
                sequence,
                init_values_2D,
                propagate_frequency=propagate_slider.value,
            )
        elif method == "approachextreme":
            history: list[list[tuple[float, float]]] = consensus(
                graph_approachextreme,
                sequence,
                init_values_2D,
                propagate_frequency=None,
            )

            history_propagate: list[list[tuple[float, float]]] = consensus(
                graph_approachextreme,
                sequence,
                init_values_2D,
                propagate_frequency=propagate_slider.value,
            )
        else:
            msg = (
                f"Unknown method: {method}."
                "Please choose among `mean` or `midpoint`."
            )
            raise ValueError(msg)

        # ----------------------------------------------------------------------
        # Plot results
        # ----------------------------------------------------------------------
        with plot1_frame:
            clear_output(wait=True)
            plt.figure(figsize=(4, 4))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            for pn in range(len(history[0])):
                # plot history
                history_x = [history[i][pn][0] for i in range(len(history))]
                history_y = [history[i][pn][1] for i in range(len(history))]
                plt.plot(
                    history_x,
                    history_y,
                    color="gray",
                    marker="o",
                    linestyle="dashed",
                )
            for pn in range(len(history[0])):
                # plot initial and final values over the rest
                history_x = [history[i][pn][0] for i in range(len(history))]
                history_y = [history[i][pn][1] for i in range(len(history))]
                plt.plot(
                    history_x[0:1],
                    history_y[0:1],
                    color="red",
                    marker="o",
                    linestyle="dashed",
                )
                plt.plot(
                    history_x[-2:-1],
                    history_y[-2:-1],
                    color="blue",
                    marker="o",
                    linestyle="dashed",
                )
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel("x", fontsize=16)
            plt.ylabel("y", fontsize=16)
            plt.show()

        with plot2_frame:
            clear_output(wait=True)
            plt.figure(figsize=(4, 4))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            for pn in range(len(history[0])):
                # plot history
                history_x = [
                    history[i][pn][0] for i in range(len(history_propagate))
                ]
                history_y = [
                    history[i][pn][1] for i in range(len(history_propagate))
                ]
                plt.plot(
                    history_x,
                    history_y,
                    color="gray",
                    marker="o",
                    linestyle="dashed",
                )
            for pn in range(len(history[0])):
                # plot initial and final values over the rest
                history_x = [
                    history[i][pn][0] for i in range(len(history_propagate))
                ]
                history_y = [
                    history[i][pn][1] for i in range(len(history_propagate))
                ]
                plt.plot(
                    history_x[0:1],
                    history_y[0:1],
                    color="red",
                    marker="o",
                    linestyle="dashed",
                )
                plt.plot(
                    history_x[-2:-1],
                    history_y[-2:-1],
                    color="blue",
                    marker="o",
                    linestyle="dashed",
                )
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel("x", fontsize=16)
            plt.ylabel("y", fontsize=16)
            plt.show()

        # ----------------------------------------------------------------------
        # Draw tables
        # ----------------------------------------------------------------------
        with table1_frame:
            clear_output(wait=True)
            display(HTML(render_table(history)))

        with table2_frame:
            clear_output(wait=True)
            display(HTML(render_table(history_propagate)))

        run_button.disabled = False

    # --------------------------------------------------------------------------
    # Deploy observers
    # --------------------------------------------------------------------------
    run_button.on_click(lambda _: run())

    run()
