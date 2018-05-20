# Your local computer
### Setting up
1. install pipenv https://github.com/pypa/pipenv
2. go to the cloned repository, run the following:
    1. pipenv install
    2. pipenv shell
    3. jupyter notebook
3. Open whatever jupyter notebook you want


# The cloud
### Setting up the cloud:
1. Install Google Cloud SDK from here: https://cloud.google.com/sdk/docs/
2. Connect to VM (see below) and run `/home/shared/setup.sh && source ~/.bashrc`
3. Also, run `jupyter notebook password` and set the password to group27password
### Connecting to a VM:
1. `gcloud compute ssh --zone=europe-west1-b group27instance`
2. go to shared directory: `cd /home/shared`. run jupyter-notebook: `jupyter-notebook --no-browser --port=7000`
3. Find the ip address by running `gcloud compute addresses list` locally (not on the VM)
    1. The ip is likely to be 35.233.64.20 (not sure if it ever changes)
4. Connect in your browser to <ip>:7000 to see the jupyter notebook - the password is group27password


### Starting the instance:
1. In the console in the browser, go to Compute Engine -> VM Instances;  
2. Start the group27instance
3. **Make sure to stop the instance once you finish working with it, as it's paid for**



### Stopping the instance
1. In the console in the browser, go to Compute Engine -> VM Instances;  
2. Stop the group27instance

### HELP I'M STUCK
Q: I cannot connect via ssh, it says "connection time-out"  
A: Most likely you haven't started the instance, see _Starting the instance_
  
Q: `jupyter-notebook` is command not found    
A: you probably haven't run the setup scripts in /home/shared, see _Setting up_



