# this is one of the cached base images available for ACI
FROM python:3.9

# Install libraries and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      zlib1g-dev \
     swig \
     && rm -rf /var/lib/apt/lists/*

# Set up the simulator
WORKDIR /sim

# Copy simulator files to /sim
COPY ./ /sim

# Install simulator dependencies
RUN pip3 install -r requirements.txt

# This will be the command to run the simulator
CMD ["python", "main.py"]