# ==============================================================================
# Utils
# ==============================================================================
OUTDIR-FIGURES=out-figures
OUTDIR-VIDEOS=out-videos

FIGURES := \
	$(OUTDIR-FIGURES)/fig1a-below.pdf \
	$(OUTDIR-FIGURES)/fig2a-gradient.pdf \
	$(OUTDIR-FIGURES)/fig4b-differential_evolution.pdf \
	$(OUTDIR-FIGURES)/fig4d-particle_swarm.pdf \
	$(OUTDIR-FIGURES)/fig5b-consensus_mean.pdf \
	$(OUTDIR-FIGURES)/fig7a-crn-abc-deterministic.pdf \
	$(OUTDIR-FIGURES)/fig8c-mis_final.pdf

VIDEOS := \
	$(OUTDIR-VIDEOS)/consensus-midextremes_video.mp4 \
	$(OUTDIR-VIDEOS)/mis_video.mp4 \
	$(OUTDIR-VIDEOS)/differential_evolution.mp4 \
	$(OUTDIR-VIDEOS)/particle_swarm.mp4 \
	$(OUTDIR-VIDEOS)/gradient_descent_momentum.mp4

# ==============================================================================
# MAIN Targets
# ==============================================================================
all: figures videos

figures: $(FIGURES)

videos: $(VIDEOS)

.PHONY: all figures videos clean

clean:
	rm -rf $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)

# Ensure output directories exist
$(OUTDIR-FIGURES) $(OUTDIR-VIDEOS):
	mkdir -p $@

# ==============================================================================
# FIGURES
# ==============================================================================
$(OUTDIR-FIGURES)/fig1a-below.pdf: src/hodgkin_huxley.py
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/hodgkin_huxley.py $(OUTDIR-FIGURES)

$(OUTDIR-FIGURES)/fig2a-gradient.pdf: src/gradient_descent.py | $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)
	python3 src/gradient_descent.py $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)

$(OUTDIR-FIGURES)/fig4b-differential_evolution.pdf: src/differential_evolution.py | $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)
	python3 src/differential_evolution.py $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)

$(OUTDIR-FIGURES)/fig4d-particle_swarm.pdf: src/particle_swarm.py | $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)
	python3 src/particle_swarm.py $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)

$(OUTDIR-FIGURES)/fig5b-consensus_mean.pdf: src/consensus.py | $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/consensus.py $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)

$(OUTDIR-FIGURES)/fig7a-crn-abc-deterministic.pdf: src/chemical_reaction_network.py
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/chemical_reaction_network.py $(OUTDIR-FIGURES)

$(OUTDIR-FIGURES)/fig8c-mis_final.pdf: src/maximal_independent_set.py | $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)
	python3 src/maximal_independent_set.py $(OUTDIR-FIGURES) $(OUTDIR-VIDEOS)

# ==============================================================================
# VIDEOS
# ==============================================================================
#
$(OUTDIR-VIDEOS)/consensus-midextremes_video.mp4: $(OUTDIR-FIGURES)/fig5b-consensus_mean.pdf | $(OUTDIR-VIDEOS)
	ffmpeg -framerate 2 -i $(OUTDIR-VIDEOS)/fig6a-midextremes/fig6a-midextremes_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $@

$(OUTDIR-VIDEOS)/mis_video.mp4: $(OUTDIR-FIGURES)/fig8c-mis_final.pdf | $(OUTDIR-VIDEOS)
	ffmpeg -framerate 2 -i $(OUTDIR-VIDEOS)/fig8-mis-rounds/fig8-mis_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEOS)/mis_video.mp4

$(OUTDIR-VIDEOS)/gradient_descent_momentum.mp4: $(OUTDIR-FIGURES)/fig2a-gradient.pdf | $(OUTDIR-VIDEOS)
	ffmpeg -framerate 3 -i $(OUTDIR-VIDEOS)/fig2a-gradient/fig2a-gradient_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEOS)/gradient_descent.mp4
	ffmpeg -framerate 3 -i $(OUTDIR-VIDEOS)/fig2b-gradient_momentum/fig2b-gradient_momentum_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEOS)/gradient_descent_momentum.mp4
	ffmpeg -framerate 3 -i $(OUTDIR-VIDEOS)/fig2c-gradient_noise/fig2c-gradient_noise_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEOS)/gradient_descent_noise.mp4

$(OUTDIR-VIDEOS)/differential_evolution.mp4: $(OUTDIR-FIGURES)/fig4b-differential_evolution.pdf | $(OUTDIR-VIDEOS)
	ffmpeg -framerate 2 -i $(OUTDIR-VIDEOS)/fig4b-differential_evolution/fig4b-differential_evolution_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEOS)/differential_evolution.mp4

$(OUTDIR-VIDEOS)/particle_swarm.mp4: $(OUTDIR-FIGURES)/fig4d-particle_swarm.pdf | $(OUTDIR-VIDEOS)
	ffmpeg -framerate 4 -i $(OUTDIR-VIDEOS)/fig4d-particle_swarm/fig4d-particle_swarm_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEOS)/particle_swarm.mp4
