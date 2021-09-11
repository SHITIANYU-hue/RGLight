# RGLight
This repository contains code provided alongside the paper : Improving the generalizability and robustness of large-scale traffic signal control


#Overview
To install the conda environment, please run:  conda env create -f environment.yml


## Overview

If you are interested in reproducing some of the results from the paper, train the model by running:

python main.py --config-dir=config/binary/GCN/Q_L/Train/DIVERSIFIED_NETS/Train_veh.ini 


This repository contains pre-trained models and parameters as well as config files ('.ini files') required to re-run(/train/evaluate) experiments included in the  paper.

The trained models: models_paramsk0 corresponds to DGRL as proposed in the paper and models_paramsk1 correspoinds to IGRL as shown in the paper.

#Run experiment

To evaluate all the methods, by running:

To evaluate the all the methods, by running:


python main.py --config-dir=config/binary/EXP1/EXP1_all.ini

To evaluate the proposed method, by running:
python main.py --config-dir=config/binary/EXP1/EXP1_veh.ini

To try different traffic flow, please change the "period" params in .ini file

To try different non-grid network structures, please set grid=False, and change num_edges_random_net params in .ini file to try different random edges

To try different grid network structure, please set grid=True, and change the col_num or row_num in .ini file to try different scale of grid network. 


#Experiment results

Results are stored alongside the .ini files. It will be recorded under /tensorboard folder.

  - For experiments involving training, these logs include training metrics (losses, reward, etc.)
  - For experiments involving evaluation, these logs include aggregated performance metrics (queue_lengths, delays, CO2 emissions, etc.)

For experiments involving evaluations, .pkl files include detailed per-trip information (delay, duration, etc.)

