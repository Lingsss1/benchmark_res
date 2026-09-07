FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 1000 benchmark \
    && useradd --uid 1000 --gid benchmark --create-home benchmark

WORKDIR /opt/recevobench
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY schemas ./schemas
COPY tasks ./tasks
COPY faults ./faults
COPY seed ./seed
RUN python -m pip install .

USER benchmark
WORKDIR /benchmark/workspace
ENTRYPOINT ["recevobench"]
