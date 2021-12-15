.ONESHELL:
SHELL=/bin/bash
ENV_NAME=plenoxel
UNAME := $(shell uname)
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

install: update-conda install-pip

update-conda:
	conda env update -f environment.yml

install-pip:
	$(CONDA_ACTIVATE) $(ENV_NAME)
	pip install .

download-tt:
	gsutil -m cp -r gs://data.netdron.es/TanksAndTempleBG.tar.gz data
	tar -xvf data/*.gz
	rm data/*.gz
