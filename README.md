### Setup
0. Clone the repository
1. export your path (path is the path to the .json file):
    1. for windows: `$env:GOOGLE_APPLICATION_CREDENTIALS="[PATH]"` in powershell or `set GOOGLE_APPLICATION_CREDENTIALS=[PATH]` in cmd
    2. for linux: `export GOOGLE_APPLICATION_CREDENTIALS="[PATH]"`
2. install Google Cloud SDK from here: https://cloud.google.com/sdk/docs/
3. Setup virtual environment, e.g. using `pipenv`  (`pip install pipenv`) - works on both windows and linux. See https://docs.pipenv.org/ for details
4. Activate the environment using `pipenv install` and `pipenv shell` **in the directory of the repo**
5. Check that everything is working by doing `python -V` and `gcloud ml-engine models list`
6. More information: https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction



### Connecting to a VM:
1. `gcloud compute ssh --zone=europe-west1-b group27instance`
2. only once: `/home/shared/setup.sh && source ~/.bashrc`
3. go to shared directory: `cd /home/shared`. run jupyter-notebook: `jupyter-notebook --no-browser --port=7000`
4. Find the static ip here: https://console.cloud.google.com/compute/instances?project=group-27-203507
