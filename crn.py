from turtle import pos
from mobspy import BaseSpecies, Simulation, Zero
import matplotlib.pyplot as plt

A, B, C = BaseSpecies()
A(10) + B(10) >> A + C[0.1]
A >> Zero[0.1]

MySim = Simulation(A | B | C)


# stochastic
MySim.run(
    simulation_method="stochastic",
    duration=10,
    save_data=False,
    repetitions=10,
    plot_data=False,
    seeds=range(10),
)
plt.figure(figsize=(4, 4))
for i, res in enumerate(MySim.results):
    plt.plot(
        res["Time"],
        res["A"],
        label="A" if i == 0 else "_",
        color="red",
        alpha=1 if i == 0 else 0.07,
    )
    plt.plot(
        res["Time"],
        res["B"],
        label="B" if i == 0 else "_",
        color="blue",
        alpha=1 if i == 0 else 0.07,
    )
    plt.plot(
        res["Time"],
        res["C"],
        label="C" if i == 0 else "_",
        color="black",
        alpha=1 if i == 0 else 0.07,
    )
plt.legend(frameon=False, loc=(0.8, 0.7))
plt.ylabel("count")
plt.xlabel("time")
plt.savefig("crn-abc-stoch.pdf", bbox_inches="tight", transparent=True, pad_inches=0)

# deterministic
MySim.run(
    simulation_method="deterministic", duration=10, save_data=False, plot_data=False
)
plt.figure(figsize=(4, 4))
for i, res in enumerate(MySim.results):
    plt.plot(
        res["Time"],
        res["A"],
        label="A" if i == 0 else "_",
        color="red",
        alpha=1 if i == 0 else 0.07,
    )
    plt.plot(
        res["Time"],
        res["B"],
        label="B" if i == 0 else "_",
        color="blue",
        alpha=1 if i == 0 else 0.07,
    )
    plt.plot(
        res["Time"],
        res["C"],
        label="C" if i == 0 else "_",
        color="black",
        alpha=1 if i == 0 else 0.07,
    )
plt.legend(frameon=False, loc=(0.8, 0.7))
plt.ylabel("count/volume")
plt.xlabel("time")
plt.savefig("crn-abc-det.pdf", bbox_inches="tight", transparent=True, pad_inches=0)
