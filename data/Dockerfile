FROM python:3

RUN apt-get -yqq update \
    && apt-get install -yq python3-dev libfftw3-dev libeigen3-dev pybind11-dev \
    && pip install --no-cache-dir click dumb-init numpy psutil tqdm

# Install GalSim with an increased cache size of 1000
RUN git clone --single-branch --branch sersic-cache https://github.com/dvukolov/GalSim.git \
    && cd GalSim \
    && python setup.py install

RUN useradd --create-home --home-dir /app docker
RUN mkdir /data
USER docker

ADD generator.py /app/
WORKDIR /data
VOLUME /data

ENTRYPOINT ["dumb-init", "--", "python", "/app/generator.py"]
CMD ["--help"]
