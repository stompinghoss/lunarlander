FROM python:3.10-slim

RUN python --version
RUN pip install --upgrade pip
RUN pip --version
RUN apt-get update && apt-get install -y swig
RUN pip install --no-cache-dir agbenchmark
# Other customization steps
