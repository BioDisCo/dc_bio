all: hh_after.pdf gradient.pdf gradient_momentum.pdf gradient_noise.pdf particle.pdf de.pdf particle_step.pdf de_step.pdf crn-alg-det.pdf crn-abc-det.pdf

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

clean:
	rm *.pdf