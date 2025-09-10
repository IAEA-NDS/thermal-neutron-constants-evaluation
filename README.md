# Thermal Neutron Constants Evaluation

**This evaluation is work in progress**

## Setup

Download the TNC evaluation pipeline:

```
git clone --recurse-submodules https://github.com/iaea-nds/thermal-neutron-constants-evaluation.git
```

Set up the virtual environment:

```
cd thermal-neutron-constants-evaluation
python -m venv venv
source venv/bin/activate
```

Install gmapy and other dependencies (still in the root dir of the repo):

```
pip install gmapy/
pip install matplotlib
pip install lark
```

## Running the pipeline 

To run the pipeline (being in the root dir of the repo):

```
cd evaluation
python tnc_inference.py
```
