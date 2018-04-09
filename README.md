# Enhancing the consistency of spaceborne (SR) and ground-based radar (GR) comparisons by using quality filters

by Irene Crisologo, Robert Warren, Kai Mühlbauer, and Maik Heistermann

This repository contains the workflow script of the SR-GR comparison enhanced by using quality filters. To test the scripts, it is necessary to install [wradlib](https://github.com/wradlib/wradlib).

The workflow is divided into three parts:

1. [Generating the beam blockage fraction (BBF) map and the corresponding quality map (Q_BBF)](https://github.com/wradlib/radargpm-beamblockage/blob/master/scripts/00_Beam_Blockage_Map.ipynb)

2. SR-GR matching ([for TRMM](https://github.com/wradlib/radargpm-beamblockage/blob/master/scripts/01_Workflow_TRMM.ipynb) | [for GPM](https://github.com/wradlib/radargpm-beamblockage/blob/master/scripts/02_Workflow_GPM.ipynb)) incorporating the use of Q_BBF map in assigning a quality value for each matched point. 

3. [Comparison and analysis of matched points.](https://github.com/wradlib/radargpm-beamblockage/blob/master/scripts/03_Overpass_analysis.ipynb) To switch between TRMM and GPM analysis, change the variable `platf` under section 3 to `trmm` or `gpm`.
