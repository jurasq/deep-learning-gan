

### Connecting to a VM:
1. `gcloud compute ssh --zone=europe-west1-b group27instance`
2. go to shared directory: `cd /home/shared`. run jupyter-notebook: `jupyter-notebook --no-browser --port=7000`
3. Find the ip address by running `gcloud compute addresses list` locally (not on the VM)
    1. The ip is likely to be 35.233.64.20 (not sure if it ever changes)
4. Connect in your browser to <ip>:7000 to see the jupyter notebook - the password is group27password


### Starting the instance:
1. In the console in the browser, go to Compute Engine -> VM Instances;  
2. Start the group27instance



### Stopping the instance
1. In the console in the browser, go to Compute Engine -> VM Instances;  
2. Stop the group27instance


