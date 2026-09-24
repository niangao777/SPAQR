# SPAQR

Except for MIMIC, which is a restricted-access private dataset, the other three datasets do not need to be prepared manually. Our code will automatically download and preprocess them when training is launched.

### Environment Setup
```bash
conda create -n spaqr python=3.12
conda activate spaqr
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Training
```bash
bash scripts/SPAQR/P12.sh
```
All parameters are pre-configured in the script file.
