OUTDIR-FIGURES=out-figures
OUTDIR-VIDEOS=out-videos

all: figures videos

figures: $(OUTDIR-FIGURES)/fig1a-below.pdf $(OUTDIR-FIGURES)/fig2a-gradient.pdf $(OUTDIR-FIGURES)/fig4b-differential_evolution.pdf $(OUTDIR-FIGURES)/fig4d-particle_swarm.pdf $(OUTDIR-FIGURES)/fig5b-consensus_mean.pdf $(OUTDIR-FIGURES)/fig7a-crn-abc-deterministic.pdf $(OUTDIR-FIGURES)/fig8c-mis_final.pdf

videos: $(OUTDIR-VIDEOS)/mis_video.mp4 $(OUTDIR-VIDEOS)/differential_evolution.mp4 $(OUTDIR-VIDEOS)/particle_swarm.mp4 $(OUTDIR-VIDEOS)/gradient_descent_momentum.mp4

$(OUTDIR-FIGURES)/fig1a-below.pdf: src/hodgkin_huxley.py
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/hodgkin_huxley.py $(OUTDIR-FIGURES)

$(OUTDIR-FIGURES)/fig2a-gradient.pdf: src/gradient_descent.py
	mkdir -p $(OUTDIR-FIGURES)
	mkdir -p $(OUTDIR-VIDEOS)
	python3 src/gradient_descent.py $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)

$(OUTDIR-FIGURES)/fig4b-differential_evolution.pdf: src/differential_evolution.py
	mkdir -p $(OUTDIR-FIGURES)
	mkdir -p $(OUTDIR-VIDEOS)
	python3 src/differential_evolution.py $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)

$(OUTDIR-FIGURES)/fig4d-particle_swarm.pdf: src/particle_swarm.py
	mkdir -p $(OUTDIR-FIGURES)
	mkdir -p $(OUTDIR-VIDEOS)
	python3 src/particle_swarm.py $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)

$(OUTDIR-FIGURES)/fig5b-consensus_mean.pdf: src/consensus.py
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/consensus.py $(OUTDIR-FIGURES)

$(OUTDIR-FIGURES)/fig7a-crn-abc-deterministic.pdf: src/chemical_reaction_network.py
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/chemical_reaction_network.py $(OUTDIR-FIGURES)

$(OUTDIR-FIGURES)/fig8c-mis_final.pdf: src/maximal_independent_set.py
	mkdir -p $(OUTDIR-FIGURES)
	mkdir -p $(OUTDIR-VIDEOS)
	python3 src/maximal_independent_set.py $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)

$(OUTDIR-VIDEOS)/mis_video.mp4: $(OUTDIR-FIGURES)/fig8c-mis_final.pdf
	mkdir -p $(OUTDIR-VIDEOS)
	ffmpeg -framerate 2 -i $(OUTDIR-VIDEOS)/fig8-mis-rounds/fig8-mis_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEOS)/mis_video.mp4

$(OUTDIR-VIDEOS)/gradient_descent_momentum.mp4: $(OUTDIR-FIGURES)/fig2a-gradient.pdf
	mkdir -p $(OUTDIR-VIDEOS)
	ffmpeg -framerate 3 -i $(OUTDIR-VIDEOS)/fig2a-gradient/fig2a-gradient_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEOS)/gradient_descent.mp4
	ffmpeg -framerate 3 -i $(OUTDIR-VIDEOS)/fig2b-gradient_momentum/fig2b-gradient_momentum_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEOS)/gradient_descent_momentum.mp4
	ffmpeg -framerate 3 -i $(OUTDIR-VIDEOS)/fig2c-gradient_noise/fig2c-gradient_noise_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEOS)/gradient_descent_noise.mp4

$(OUTDIR-VIDEOS)/differential_evolution.mp4: $(OUTDIR-FIGURES)/fig4b-differential_evolution.pdf
	mkdir -p $(OUTDIR-VIDEOS)
	ffmpeg -framerate 2 -i $(OUTDIR-VIDEOS)/fig4b-differential_evolution/fig4b-differential_evolution_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEOS)/differential_evolution.mp4

$(OUTDIR-VIDEOS)/particle_swarm.mp4: $(OUTDIR-FIGURES)/fig4d-particle_swarm.pdf
	mkdir -p $(OUTDIR-VIDEOS)
	ffmpeg -framerate 4 -i $(OUTDIR-VIDEOS)/fig4d-particle_swarm/fig4d-particle_swarm_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEOS)/particle_swarm.mp4

clean:
	rm -r $(OUTDIR-FIGURES)/*.pdf $(OUTDIR-FIGURES)/*/*.pdf $(OUTDIR-VIDEOS)/*/*.png $(OUTDIR-VIDEOS)/*.mp4; \
	rmdir $(OUTDIR-FIGURES)/* $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)/* $(OUTDIR-VIDEOS)
