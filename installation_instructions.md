## Preparation:
- Install JupyterLab or Jupyter Notebook: https://jupyter.org/install 
- Clone this repository: ```git clone git@github.com:shlevanto/pygobnilp_phd.git```

## Python virtual environment with Conda
- If not already installed, install JupyterLab: ```conda install -c forge jupyterlab```
- Create environment: ```conda env create --file=conda_env.yml```
- Activate environment: ```conda activate pygobnilp```

## Activate license (contact simo.levanto@helsinki.fi):
Inside the conda environment run ```grbprobe``` 
Input the information into the Gurobi license system
Download license file
Copy license file gurobi.lic to user’s home folder

## Sanity check:
```python rungobnilp.py data/asia_10000.dat```

