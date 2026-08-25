FROM python:2.7.18-buster

RUN printf 'deb http://archive.debian.org/debian buster main\n' \
      > /etc/apt/sources.list \
    && printf 'Acquire::Check-Valid-Until "false";\n' \
      > /etc/apt/apt.conf.d/99archive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         ca-certificates \
         git \
         g++ \
         gfortran \
         libblas-dev \
         liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-legacy.txt /tmp/requirements-legacy.txt

RUN pip install --no-cache-dir \
      -r /tmp/requirements-legacy.txt