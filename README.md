# Executive-Comp-to-Patient-Care
Investigates the correlational relationship between executive incentives (financial structures) and hospital-level strategic decisions

# Running the analysis

## Installation
The most straightforward way to run the notebooks are either through a virtual environment or a devcontainer. 

Regardless of installation method, there is one dataset that must be manually downloaded by following these steps:

1. Go to this URL: https://www.ahrq.gov/chsp/data-resources/compendium-2022.html
2. Download the 'Hospital Linkage File' as a csv.
3. Rename it to 'health_system.csv'
4. place it in the `./data/` folder

### Setting up the project using a virtual environment 
1. `python -m venv env`
2. `source ./env/bin/activate` (macOs and Linux) or `source ./env/Scripts/activate` (windows using git-bash)
3. `pip install -r requirements.txt`
4. `jupyter lab`

### Setting up the project using devcontainers
I do not have vscode so I run everything through the devcontainer CLI (https://github.com/devcontainers/cli)

1. `devcontainer up --id-label "exec-comp" --workspace-folder .` (creates the containers)
2. `devcontainer exec --id-label "exec-comp" --workspace-folder . jupyter lab --allow-root` (enters the container and starts the notebook)

## Project structure 

There are 5 jupyter notebooks that run the project `data_preparation.py` and `project_utils.py` contain helper functions to make the notebooks more readable. 

`load/`: destination directory for downloaded zip files.

`data/`: destination for unzipped, but unprocessed files (e.g. Extracted 990 xml documents).

`processed/`: destination for cleaned data

`EDA/`: contains figures from EDA notebook.

`results/`: figures and tables from regression and correlation analysis. 

`archive/` contains old notebooks that are not immediately needed, but contain old versions of the data pipeline that may be useful should this project need to be updated.


## Notebooks

To reproduce the project, run each of these jupyter notebooks in order.

### 1. Download datasets.ipynb

Run this to download all required datasets. This should take up ~30GB and about 20 minutes to download.

### 2. Clean 990 dataset.ipynb 
Convert 990 forms from xml filer to csv files. Please note that this may take 60+ minutes depending on your hardware

### 3. EDA.ipynb 
Generates some basic figures (e.g. histograms)

### 4. Regression.ipynb
Run all the regression models used in the project. Results are saved to `./results/`

# Datasets

The following datasets are used in this project:
1. IRS 990 dataset
    a. link to dataset: https://www.irs.gov/statistics/soi-tax-stats-annual-extract-of-tax-exempt-organization-financial-data
    b. field names for form 990: https://www.irs.gov/pub/irs-tege/2022form990withfieldnames.pdf
2. CMS hospital dataset: https://data.cms.gov/provider-data/archived-data/hospitals
    a. This project uses hospitals_10_2022.zip
3. CMS Hospital cost report: https://data.cms.gov/provider-compliance/cost-report/hospital-provider-cost-report
4. Health system compendium: https://www.ahrq.gov/chsp/data-resources/compendium-2022.html
5. Linkage file mapping 990 EIN to CMS: https://www.communitybenefitinsight.org/?page=info.data_api
6. state to region map: https://www.kaggle.com/datasets/omer2040/usa-states-to-region


