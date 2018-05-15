

### Connecting to a VM:
1. `gcloud compute ssh --zone=europe-west1-b group27instance`
2. go to shared directory: `cd /home/shared`. run jupyter-notebook: `jupyter-notebook --no-browser --port=7000`
3. Find the ip address by running `gcloud compute addresses list`
4. Connect to <ip>:7000 - the password is group27password


### Starting the instance:
1. In the console in the browser, go to Compute Engine -> VM Instances;  
2. Start the group27instance



### Stopping the instance
1. In the console in the browser, go to Compute Engine -> VM Instances;  
2. Stop the group27instance


