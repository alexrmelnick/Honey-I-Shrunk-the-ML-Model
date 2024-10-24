# Getting started

## New Instructions for SCC

1. Run `module load miniconda`
1. Run `conda load activate fall-2024-pyt`
1. Navigate to the directory with `cd /projectnb/ece601/Honey_I_Shrunk_the_ML_Model/Honey-I-Shrunk-the-ML-Model`. You are now in our project directory.
1. Run `git pull` to get the latest changes.


## Old Instructions (We probably want to delete this at some point)

**PLEASE JUST USE LINUX** Please note that this install can take more than an hour depending on download speeds.

1. Install SSL dependencies using `sudo apt install build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git`.
1. Install Python 3.9.9 from [python.org](https://www.python.org/downloads/release/python-399/). Run `sudo tar xzf Python-3.9.9.tgz` to extract the tarball, then `cd Python-3.9.9` and `./configure` to configure the build. Run `sudo make altinstall` to install the Python executable as an alternate version.
1. Install CUDA (If CUDA 12.4 supports your current version of Linux, install that one. Otherwise install the latest version of CUDA that supports your Linux Version) from [nvidia.com](https://developer.nvidia.com/cuda-downloads). Follow the provided instructions to install the CUDA toolkit. **Make sure to finish the instructions after the long download!**
1. Install FFmpeg with `sudo apt install ffmpeg`.
1. **SKIP THIS UNLESS WE ARE CHANGING THE INSTALLATION INSTRUCTIONS** If you are redoing the virtual environment, run `rm -rf myenv` to remove the old virtual environment. Then run `virtualenv -p python3.9.9 myenv` to create a new virtual environment.
1. Run `pip install virtualenv` to install the virtual environment package.
1. Run `myenv\Scripts\activate` on Windows or `source myenv/bin/activate` on MacOS/Linux to activate the virtual environment.
1. Run `pip install -r requirements.txt` to install the required packages. Note that this will require ~1GB of disk space.
1. *If you are using Whisper for the first time*, run `whisper --model tiny.en "Whisper_Test.mp3"` to download and test the model. You should see the text of the speech in the terminal. Feel free to ignore any errors that occur as long as the mp3 is transcribed.
1. Do your work.
1. If you want to add a new package, run `pip install <package-name>` and then run `pip freeze > requirements.txt` to update the requirements file.
1. After you are done, run `deactivate` to deactivate the virtual environment.
