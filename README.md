# DMID-Analysis-Music-Dynamics
This is the github repo for DMID-Deep Music Information Dynamics.


# For Fugue and Prelude VAE

Python dependencies for the audio experiments are listed in "Fugue and Prelude VAE"/requirements.txt. To install these dependencies, run
```
pip install -r "Fugue and Prelude VAE"/requirements.txt
```
Run the jupyter notebook "fugue_prelude_vae.ipynb", it contains comments in cells that guide you into all the training and evaluation parts.

# For Bach Invention VAE

We split the training and evaluation of "Bach Invention VAE" into two jupyter notebooks. The reason is because the VAE structure for this part is partially referred from [Music SketchNet](https://github.com/RetroCirce/Music-SketchNet), and [Music InpaintNet](https://github.com/ashispati/InpaintNet). We split two parts to make the pipeline clear.

Python dependencies for the audio experiments are listed in "Bach Invention VAE"/requirements.txt. To install these dependencies, run
```
pip install -r "Bach Invention VAE"/requirements.txt
```

Run the code by following steps:
1. Train the MeasureVAE by running the code "train_measurevae.ipynb", or you can directly use our saved model in the model_backup directory.
2. Run the "evaluation_bach.ipynb" for different settings of the experimenets. The comment in the jupyter notebook will help you reproduce the whole experiment as reported in the paper.


# For Audio VAE
Python dependencies for the audio experiments are listed in "Prelude Audio VAE"/requirements.txt. To install these dependencies, run
```
pip install -r "Prelude Audio VAE"/requirements.txt
```
The dataset used can be downloaded from [here](http://labrosa.ee.columbia.edu/projects/piano/).

Run the whole code in the "Prelude Audio VAE"/vea_audio_zmu_oracle_vge.ipynb. The notebook will guide you through all the training and evaluation parts.
