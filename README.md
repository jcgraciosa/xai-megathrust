# xai-megathrust
Repository for the codes used in the paper: Testing driving mechanisms of megathrust seismicity with Explainable Artificial Intelligence.

Juan Carlos Graciosa, Fabio A. Capitanio, Adam Beall, Mitchell Hargreaves, Thyagarajulu Gollapalli, Titus Tang, Mohd Zuhair

## Directories:
1. helper_pkg: Package containing helper routines used during the creation of grids.
2. in-data: Contains the processed but non-standardized features. Standardization is done during runtime.
3. ml4szeq: Main set of codes used in the study.
4. ntbk: Notebooks used in the study. This includes the sampling of the raw data into grids (0_grid_sampling.ipynb), creation of classification maps (1_make_classification_maps.ipynb), and the creation of LRP heatmaps (2_make_lrp_heatmaps.ipynb).
5. vis_pkg: Package used for creating maps

## Raw dataset:
The unprocessed dataset are found in: 10.26180/25066592.
The data prefix indicates the convergent region it is a part of and are as follows:

1. alu: Alaska-Aleutians
2. cam: Central America
3. izu: Izu-Bonin-Mariana
4. ker: Tonga-Kermadec
5. kur: Japan-Kuriles-Kamchatka
6. ryu: Ryukyu-Nankai
7. sam: South America
8. sum: Southeast Asia 

This was adapted from the notation used by Hayes et al., 2018.
