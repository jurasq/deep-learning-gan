### Setup
0. Clone the repository
1. export your path (path is the path to the .json file):
    1. for windows: `$env:GOOGLE_APPLICATION_CREDENTIALS="[PATH]"` in powershell or `set GOOGLE_APPLICATION_CREDENTIALS=[PATH]` in cmd
    2. for linux: `export GOOGLE_APPLICATION_CREDENTIALS="[PATH]"`
2. install Google Cloud SDK from here: https://cloud.google.com/sdk/docs/
3. Setup virtual environment, e.g. using `pipenv`  (`pip install pipenv`) - works on both windows and linux. See https://docs.pipenv.org/ for details
4. Activate the environment using `pipenv install` and `pipenv shell` *in the directory of the repo*
5. Check that everything is working by doing `python -V` and `gcloud ml-engine models list`
6. More information: https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction
