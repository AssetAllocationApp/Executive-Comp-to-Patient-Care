# Executive-Comp-to-Patient-Care
Investigates the correlational relationship between executive incentives (financial structures) and hospital-level strategic decisions

# Running the analysis

## Installation
The most straightforward way to run the notebooks are either through a virtual environment or a devcontainer. 

First, there is one dataset that must be manually downloaded: 

1. Go to this URL: https://www.ahrq.gov/chsp/data-resources/compendium-2022.html
2. Download the 'Hospital Linkage File' as a csv.
3. Rename it to 'health_system.csv'
4. place it in the `./data/` folder

### Virtual environment

1. `python -m venv env`
2. `source ./env/bin/activate` (macOs and Linux) or `source ./env/Scripts/activate` (windows using git-bash)
3. `pip install -r requirements.txt`
4. `jupyter lab`

### Dev Container
I do not have vscode so I run everything through the devcontainer CLI (https://github.com/devcontainers/cli)

1. `devcontainer up --id-label "exec-comp" --workspace-folder .`
2. `devcontainer exec --id-label "exec-comp" --workspace-folder . jupyter lab`

## Notebooks
**1. Download datasets.ipynb** Run this to download all required datasets. This should take up ~30GB

**2. Clean 990 dataset.ipynb** Convert 990 forms from xml filer to csv files

**3. EDA.ipynb** Some basic data analysis 

**4. Regression.ipynb** Run all the regression models used in the project. Results are saved to `./results/`

# Datasets

All datasets are automatically downloaded except for state-to-region map (included in this repo) and the hospital compendium file, which must be downloaded manually. 
Manually Download the hospital systems compendium file.

