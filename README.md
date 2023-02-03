# Notes to make into docs

### How to make a python sim

1. Test by normal testing stuff (depending on your sim)
2. Test by running main.py (unmanaged sim)
3. Test by running docker container (unmangaged sim)
    - `docker build -t test-sim .`
    - `docker run --env SIM_ACCESS_KEY=$SIM_ACCESS_KEY --env SIM_WORKSPACE=$SIM_WORKSPACE test-sim sh`
4. 2 ways to upload managed sim:
    i. Simple way (drag and drop):
        - Go to your Bonsai workspace
        - Click "Add Sim" under simulators
        - Click on the "Python" one
        - Compress the sim into a zip file
        - Drag and drop that zip file
    ii. "Full" way using Docker
