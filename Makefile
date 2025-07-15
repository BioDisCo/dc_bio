OUTDIR=out-figures

all: figures crn-alg-$(OUTDIR)/det.pdf crn-abc-$(OUTDIR)/det.pdf mis_video.mp4

figures: $(OUTDIR)/fig1a-below.pdf $(OUTDIR)/fig2a-gradient.pdf $(OUTDIR)/fig4b-differential_evolution.pdf $(OUTDIR)/fig4d-particle_swarm.pdf $(OUTDIR)/fig5b_consensus_mean.pdf

$(OUTDIR)/fig1a-below.pdf: src/hodgkin_huxley.py
	mkdir -p $(OUTDIR)
	python3 src/hodgkin_huxley.py $(OUTDIR)

$(OUTDIR)/fig2a-gradient.pdf: src/gradient_descent.py
	mkdir -p $(OUTDIR)
	python3 src/gradient_descent.py $(OUTDIR)

$(OUTDIR)/fig4b-differential_evolution.pdf: src/differential_evolution.py
	mkdir -p $(OUTDIR)
	python3 src/differential_evolution.py $(OUTDIR)

$(OUTDIR)/fig4d-particle_swarm.pdf: src/particle_swarm.py
	mkdir -p $(OUTDIR)
	python3 src/particle_swarm.py $(OUTDIR)

$(OUTDIR)/fig5b_consensus_mean.pdf: src/consensus.py
	mkdir -p $(OUTDIR)
	python3 src/consensus.py $(OUTDIR)

crn-alg-$(OUTDIR)/crn_anihilation.pdf: crnalg.py
	mkdir -p $(OUTDIR)
	python3 src/crnalg.py $(OUTDIR)

crn-abc-$(OUTDIR)/det.pdf: crn.py
	mkdir -p $(OUTDIR)
	python3 src/crn.py $(OUTDIR)

mis-$(OUTDIR)/final.pdf: mis.py
	mkdir -p $(OUTDIR)
	python3 src/mis.py $(OUTDIR)

mis_video.mp4: mis-$(OUTDIR)/final.pdf
	mkdir -p $(OUTDIR)
	ffmpeg -framerate 2 -i mis-%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $(OUTDIR)/mis_video.mp4

clean:
	rm $(OUTDIR)/*.pdf $(OUTDIR)/*.png $(OUTDIR)/*.mp4; \
	rmdir $(OUTDIR)