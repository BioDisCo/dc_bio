# Distributed Computing Inspired by Biology

Repository containing all scripts and demo notebooks associated with the submitted paper:
> Függer, M., Nowak, T., Thuillier, K. (2025). Distributed Computing Inspired by Biology. Seminars in Cell and Developmental Biology.


## Requirements

Python 3 requirements are listed in `requirements.txt`.
A virtual Python 3 environment with all dependencies can be installed with:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Jupyter Notebook

We provide a *jupyter notebook* (`notebook.ipynb`) to showcase the bio-inspired algorithms discussed in the associated paper.

It can be visualized at https://nbviewer.org/github/BioDisCo/dc_bio/tree/main/.

It can also be executed online (*without any installation*), using *Binder* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BioDisCo/dc_bio/HEAD).


## Figures and Videos

Use `make` to generate all figures and supplementary videos.

>***Remarks:*** Generated figures may differ from the paper ones.
Indeed, the scripts heavily rely on `numpy.random` and `random` packages.
While, we define fixed seeds for everything, the behaviors of these pseudo-random algorithms may changed depending on the user OS, and the OS and packages versions.

>***Warning:*** all Python requirements should be installed.

### Paper's Figures

*Python 3* scripts generating the associated paper figures are stored in `src`.
All figures can be generated using:
```bash
make figures
```
Figures will be generated in the `out-figures` directory.

### Supplementary Videos

`ffmeg` is required to generated the supplementary videos.
They can be generated using:
```bash
make videos
```
Videos will be generated in the `out-videos` directory.
