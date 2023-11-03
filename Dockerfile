# golob/make_cags:0.0.2
FROM --platform=amd64 python:3.11-slim-buster

WORKDIR /src/
COPY . .

RUN python3 -m pip install -U pip wheel
RUN python3 -m pip install .