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

COPY requirements-baselines.txt /sim

# Install simulator dependencies
RUN pip3 install -r requirements-baselines.txt

COPY extract_nn.py moab_env.py moab_sim.py policies.py run-ff.py run-lstm.py /sim
# Add this after since you change the experiments in train.py
COPY train.py /sim

# This will be the command to run the training
CMD ["python", "train.py"]
