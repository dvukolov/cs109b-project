FROM continuumio/miniconda3

RUN conda install -yq matplotlib \
    && conda install -yq -c conda-forge galsim \
    && conda clean -yq --all \
    && pip install dumb-init streamlit

RUN useradd --create-home --home-dir /app docker
USER docker

ADD . /app/
WORKDIR /app

ENTRYPOINT ["dumb-init", "--"]
CMD streamlit run --server.port $PORT --server.headless true app.py