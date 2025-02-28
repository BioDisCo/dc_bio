all: hh_before.pdf hh_after.pdf gradient.pdf gradient_momentum.pdf gradient_noise.pdf particle.pdf de.pdf particle_step.pdf de_step.pdf crn-alg-det.pdf crn-alg-stoch.pdf crn-abc-det.pdf crn-abc-stoch.pdf mis_video.mp4

%.pdf: %.py
	python3 $<

hh_before.pdf hh_after.pdf: hh.py
	python3 $<

crn-alg-det.pdf crn-alg-stoch.pdf: crnalg.py
	python3 $<

crn-abc-det.pdf crn-abc-stoch.pdf: crn.py
	python3 $<

mis-final.pdf mis-000.png: mis.py
	python3 $<

mis_video.mp4: mis-000.png
	ffmpeg -framerate 2 -i mis-%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p $@

clean:
	rm *.pdf *.png mis_video.mp4

.PHONY: all clean
