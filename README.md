# Towards General Purpose Vision Systems
By [Tanmay Gupta](http://tanmaygupta.info/), [Amita Kamath](https://nlp.stanford.edu/~kamatha/), [Aniruddha Kembhavi](https://anikem.github.io/), and [Derek Hoiem](https://dhoiem.cs.illinois.edu/)

![teaser](assets/teaser.png)

# Overview
This code base contains the extension of GPV-1 to GPV-2 using the T5 + VinVL model.
To clone the repository use:

```
git clone --recurse-submodules git@github.com:chrisc36/gpv.git
git checkout 
```

# Installation
## Code
Create conda environment
```
conda create -n gpv python=3.6 -y
conda activate gpv
```

Next install [pytorch](https://pytorch.org/), I have been using pytorch 1.8, 
other versions might work but are not tested. For example:

```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

but you might need to change that command depending on your operating system/gpu setup.

Finally install libraries:
```bash
bash setup_conda_env.sh
```

## Data
Run:
```bash
bash setup_data.sh 
```

to download the coco data. The script assumes that source data (e.g., images and datasets) should be saved in ~/data/gpv while 
./data-cache can be used to cache things like pre-computed features. 

The web and OpenSCE dataset need to be downloaded manually at the moment.

## Set file paths
The paths in exp/ours/file_paths.py need to be modified to point to the correct locations, it
should not need to be changed if you used the default paths in setput_data.sh.

# Training
The repo is currently setup to train the basic model on COCO data, optionally with
the addition of web data.

To train on devices 0 and 1 of your machine without web data:

```angular2html

```

For debugging purposes I recommend using the --debug flag and reducing the number of devices and 
workers to 0 which will get you much faster startup times and better error messages:

```angular2html

```

which will run the model on a small sample of the data and without complicated distributed training.