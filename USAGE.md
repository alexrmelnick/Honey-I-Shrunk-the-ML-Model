# Getting started

1. Run `pip install virtualenv` to install the virtual environment package.
1. Run `myenv\Scripts\activate` on Windows or `source myenv/bin/activate` on MacOS/Linux to activate the virtual environment.
1. Run `pip install -r requirements.txt` to install the required packages. Note that this will require ~1GB of disk space.
1. Do your work.
1. If you want to add a new package, run `pip install <package-name>` and then run `pip freeze > requirements.txt` to update the requirements file.
1. After you are done, run `deactivate` to deactivate the virtual environment.
