# Setup gector for automatic evaluation of grammar errors. 
# Feel free to check the original repo: https://github.com/grammarly/gector

git clone git@github.com:jiacheng-xu/gector.git
git pull

conda create -n run-gector python=3.7
conda activate run-gector

cd gector
wget https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th
pip install -r requirements.txt
