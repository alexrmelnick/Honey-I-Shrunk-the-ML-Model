# Getting started

## New Instructions for SCC

1. Run `module load miniconda`
1. Run `conda load activate fall-2024-pyt`
1. Navigate to the directory with `cd /projectnb/ece601/Honey_I_Shrunk_the_ML_Model/Honey-I-Shrunk-the-ML-Model`. You are now in our project directory.
1. Run `git pull` to get the latest changes.

## Issues with GitHub

- GitHub has a 100MB limit on file size. If you are trying to push a commit with a file that is larger than 100MB, you will get an error. To fix this, copy all the files you have worked on to a different directory outside the repo, then run `git reset --hard HEAD` to reset the repository to the last pushed commit. Then, copy the files back into the repository and commit them again.
- We have added the "Models" folder to the .gitignore. This is because the models are too large to be pushed to GitHub. If you need to run the models, you will need to download them from the Google Drive and place them in the "Models" folder. If you are running the models on SCC, the models should already in the "Models" folder.
- Please keep up to date copies of the models on the Google Drive. If you make changes to the models, please upload the new models to the Google Drive.
