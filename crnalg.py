from turtle import pos
from mobspy import BaseSpecies, Simulation, Zero
import matplotlib.pyplot as plt

duration = 10

A, B, R = BaseSpecies()
A(12) + B(8) >> A [.1]
A + B >> B [.1]
A + R(100) >> A + A [0.01]
B + R >> B + B [0.01]

MySim = Simulation(A | B | R)


# stochastic
MySim.run(
    simulation_method="stochastic",
    duration=duration,
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
plt.gca().set_yscale('log')
plt.ylim([0.1,100])
plt.legend(frameon=False, loc=(0.8, 0.7))
plt.ylabel("count")
plt.xlabel("time")
plt.savefig("crn-alg-stoch.pdf", bbox_inches="tight", transparent=True, pad_inches=0)

# deterministic
MySim.run(
    simulation_method="deterministic", duration=duration, save_data=False, plot_data=False
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
plt.gca().set_yscale('log')
plt.ylim([0.1,100])
plt.legend(frameon=False, loc=(0.8, 0.7))
plt.ylabel("count/volume")
plt.xlabel("time")
plt.savefig("crn-alg-det.pdf", bbox_inches="tight", transparent=True, pad_inches=0)
