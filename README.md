# Distributed Computing Inspired by Biology

Repository containing all scripts and demo notebooks associated with the submitted paper:
> Függer, M., Nowak, T., Thuillier, K. (2025). Distributed Computing Inspired by Biology. Seminars in Cell and Developmental Biology.


## Requirements

Python 3 requirements are described in `requirements.txt`
```text
numpy
networkx
scipy
mobspy
ipykernel
```
A virtual Python 3 environment with all dependencies can be installed with:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Paper's Figures

*Python 3* scripts generating the associated paper figures are stored in `src`.
All figures can be generated using:
```bash
Make figures
```
Figures will be generated in the `out-figures` directory.

Supplementary videos can be generated using:
```bash
Make videos
```
`ffmeg` is required to generated the supplementary videos.

**Remarks:** Use `Make` to generate both all figures and supplementary videos.

## Notebooks

They can be visualized at https://nbviewer.org/github/BioDisCo/dc_bio/tree/main/.

They can be executed online (without any installation), using Binder [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BioDisCo/dc_bio/HEAD).