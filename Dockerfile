FROM continuumio/miniconda3

RUN mkdir -p connect-four

COPY . /connect-four
WORKDIR /connect-four

RUN apt-get update && apt-get install -y doxygen graphviz git

RUN conda env create --name connect-four --file environment.yml

RUN echo "conda activate connect-four" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pre-commit install
