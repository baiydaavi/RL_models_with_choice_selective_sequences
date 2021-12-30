# Models of decision making using choice-selective sequences

This repository contains the code used to run the Synaptic Plasticity and the Neural Dynamics model from the paper <https://www.biorxiv.org/content/10.1101/725382v3>.

## Installing Python dependencies

I strongly suggest creating a new conda environment before installing all the Python dependencies. Follow these steps to create a conda environment and install all the python dependencies:

* Create a new conda environment - ```conda create -n env_name python=3.9.4``` (replace ```env_name``` with the desired environment name).
* Activate the new conda environment - ```conda activate env_name```
* Install all the python dependencies - ```pip install -r requirements.txt```

You are now ready to run all the python scripts. Remember to install Jupyter Notebook or JupyterLab to run the notebooks.

## How to Use?

### Synaptic Plasticity Model

* To run the Synaptic Plasticity model for the various choice-selective inputs and generate the plots from the paper, run the ```plots.ipynb``` notebook located in the SynapticPlasticityModel folder.

### Neural Dynamics Model

* To generate the plots from the paper using the saved Neural Dynamics model, run the ```plots.ipynb``` notebook located in the NeuralDynamicsModel folder.

* To train the Neural Dynamics model from scratch using the default parameters, run the following command:

    ```python train.py```

* To generate testing data for the Neural Dynamics model using the default parameters, run the following command:

    ```python test.py```