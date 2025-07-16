OUTDIR-FIGURES=out-figures
OUTDIR-VIDEO=out-videos

all: figures crn-alg-$(OUTDIR-FIGURES)/det.pdf crn-abc-$(OUTDIR-FIGURES)/det.pdf mis_video.mp4

figures: $(OUTDIR-FIGURES)/fig1a-below.pdf $(OUTDIR-FIGURES)/fig2a-gradient.pdf $(OUTDIR-FIGURES)/fig4b-differential_evolution.pdf $(OUTDIR-FIGURES)/fig4d-particle_swarm.pdf $(OUTDIR-FIGURES)/fig5b-consensus_mean.pdf $(OUTDIR-FIGURES)/fig7a-crn-abc-deterministic.pdf $(OUTDIR-FIGURES)/fig8c-mis_final.pdf

videos: $(OUTDIR-VIDEO)/mis_video.mp4

$(OUTDIR-FIGURES)/fig1a-below.pdf: src/hodgkin_huxley.py
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/hodgkin_huxley.py $(OUTDIR-FIGURES)

$(OUTDIR-FIGURES)/fig2a-gradient.pdf: src/gradient_descent.py
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/gradient_descent.py $(OUTDIR-FIGURES)

$(OUTDIR-FIGURES)/fig4b-differential_evolution.pdf: src/differential_evolution.py
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/differential_evolution.py $(OUTDIR-FIGURES)

$(OUTDIR-FIGURES)/fig4d-particle_swarm.pdf: src/particle_swarm.py
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/particle_swarm.py $(OUTDIR-FIGURES)

$(OUTDIR-FIGURES)/fig5b-consensus_mean.pdf: src/consensus.py
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/consensus.py $(OUTDIR-FIGURES)

$(OUTDIR-FIGURES)/fig7a-crn-abc-deterministic.pdf: src/chemical_reaction_network.py
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/chemical_reaction_network.py $(OUTDIR-FIGURES)

$(OUTDIR-FIGURES)/fig8c-mis_final.pdf: src/maximal_independent_set.py
	mkdir -p $(OUTDIR-FIGURES)
	python3 src/maximal_independent_set.py $(OUTDIR-FIGURES)

$(OUTDIR-VIDEO)/mis_video.mp4: $(OUTDIR-FIGURES)/fig8c-mis_final.pdf
	mkdir -p $(OUTDIR-VIDEO)
	ffmpeg -framerate 2 -i $(OUTDIR-FIGURES)/fig8-mis-rounds/fig8-mis_round_%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR-VIDEO)/mis_video.mp4

clean:
	rm -r $(OUTDIR-FIGURES)/*.pdf $(OUTDIR-FIGURES)/*/*.pdf $(OUTDIR-VIDEO)/*.mp4; \
	rmdir $(OUTDIR-FIGURES)/* $(OUTDIR-FIGURES) $(OUTDIR-VIDEO)