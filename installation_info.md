## Preparation:
- Install JupyterLab or Jupyter Notebook: https://jupyter.org/install 
- Clone this repository: ```git clone git@github.com:shlevanto/pygobnilp_phd.git```

## Python modules, choose one
### A) Conda (recommended):
- If not already installed, install JupyterLab: ```conda install -c forge jupyterlab```
- Create environment: ```conda env create --file=conda_env.yml```
- Activate environment: ```conda activate pygobnilp```

### B) Python venv + pip:
- Install Graphviz: https://graphviz.org/download/
- Create virtual environment in pygobnilp folder: python -m venv venv
- Activate environment: venv\Scripts\activate (Windows)
- Install requirements: pip install -r requirements.txt
- Install pygraphviz with: ```python -m pip install --config-settings="--global-option=build_ext" --config-settings="--global-option=-IC:\Program Files\Graphviz\include" --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" pygraphviz```

## Activate license (contact simo.levanto@helsinki.fi):
Inside the conda environment run ```grbprobe``` 
Input the information into the Gurobi license system
Download license file
Copy license file gurobi.lic to user’s home folder

## Sanity check:
```python rungobnilp.py data/asia_10000.dat```

