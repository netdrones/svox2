.ONESHELL:
SHELL=/bin/bash
ENV_NAME=plenoxel
UNAME := $(shell uname)
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

install: update-conda ninja install-pip

update-conda:
	conda env update -f environment.yml

ninja:
	sudo apt install ninja-build

install-pip:
	$(CONDA_ACTIVATE) $(ENV_NAME)
	export MAX_JOBS=$$(nproc) && pip install .

download-tt:
	if [ ! -d ./data ]; then mkdir data; fi
	pushd data && gsutil -m cp -r gs://data.netdron.es/TanksAndTempleBG.tar.gz .
	tar -xvf *.gz > /dev/null
	rm *.gz && popd
