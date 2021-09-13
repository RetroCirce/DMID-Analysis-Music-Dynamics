# DMID-Analysis-Music-Dynamics
This is the github repo for DMID-Deep Music Information Dynamics.
Make sure that you install all the dependencies (mainly Pytorch) for running this code.


# For Fugue and Prelude VAE
Run the whole code in the "Fugure and Prelude VAE", it contains a jupyter notebook that guides you into all the training and evaluation parts.

# For Bach Invention VAE

1. Train the MeasureVAE by running the code "train_measurevae.ipynb", or you can directly use our saved model in the model_backup directory.
2. Run the "evaluation_bach.ipynb", for different settings of the experimenets, you need to run the cell from "Start Corruption" to "End Corruption". The comment in the jupyter notebook will help you finish the experiment.


# For Audio VAE
Python dependencies for the audio experiments are listed in "Prelude Audio VAE"/requirements.txt. To install these dependencies, run
```
pip install -r "Prelude Audio VAE"/requirements.txt
```
The dataset used can be downloaded from [here](http://labrosa.ee.columbia.edu/projects/piano/).

Run the whole code in the "Prelude Audio VAE"/vea_audio_zmu_oracle_vge.ipynb. The notebook will guide you through all the training and evaluation parts.
