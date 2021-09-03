################################################################################
# conda installs
################################################################################
conda install -c conda-forge scikit-image=0.17.2 spacy=2.3.2 spacy-lookups-data=0.3.0 -y
conda install h5py=2.10.0 -y
conda install -c cyclus java-jdk=8.45.14 -y
python -m spacy download en_core_web_sm


################################################################################
# pip installs
################################################################################
pip install -r requirements.txt