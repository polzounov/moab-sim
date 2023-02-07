# this is one of the cached base images available for ACI
FROM python:3.7.13

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

COPY requirements.txt /sim

# Install simulator dependencies
RUN pip3 install -r requirements.txt

# Copy simulator files to /sim
COPY main.py moab_interface.json moab_sim.py /sim/

# This will be the command to run the simulator
CMD ["python", "main.py"]
