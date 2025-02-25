all: hh_after.pdf gradient.pdf gradient_momentum.pdf gradient_noise.pdf particle.pdf de.pdf particle_step.pdf de_step.pdf crn-alg-det.pdf crn-abc-det.pdf mis_video.mp4

hh_after.pdf: hh.py
	python3 hh.py

gradient.pdf: gradient.py
	python3 gradient.py

gradient_momentum.pdf: gradient_momentum.py
	python3 gradient_momentum.py

gradient_noise.pdf: gradient_noise.py
	python3 gradient_noise.py

particle.pdf: particle.py
	python3 particle.py

particle_step.pdf: particle_step.py
	python3 particle_step.py

de.pdf: de.py
	python3 de.py

de_step.pdf: de_step.py
	python3 de_step.py


crn-alg-det.pdf: crnalg.py
	python3 crnalg.py

crn-abc-det.pdf: crn.py
	python3 crn.py

mis-final.pdf: mis.py
	python3 mis.py

mis_video.mp4: mis-final.pdf
	ffmpeg -framerate 2 -i mis-%03d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p mis_video.mp4

clean:
	rm *.pdf *.png mis_video.mp4