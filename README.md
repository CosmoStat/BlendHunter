![blendhunter](https://user-images.githubusercontent.com/7417573/127934298-39734525-6325-4d98-900d-136227f03b38.png)

[![PyPI pyversions](https://img.shields.io/badge/python-3.6-blue.svg)](https://python.org)

# BlendHunter
Deep learning tool for identifying blended galaxy images in survey images.

---
> Main contributor:  <a href="https://github.com/sfarrens" target="_blank" style="text-decoration:none; color: #F08080">Samuel Farrens</a>  
> Email: <a href="mailto:samuel.farrens@cea.fr" style="text-decoration:none; color: #F08080">samuel.farrens@cea.fr</a>
---


BlendHunter implements the VGG16 CNN to identify blended galaxy images in
postage stamps.

## Dependencies
The following python packages should be installed with their specific dependencies:

- [numpy](https://github.com/numpy/numpy)
- [ModOpt](https://github.com/CEA-COSMIC/ModOpt)
- [LMFIT](https://lmfit.github.io/lmfit-py/)
- [SF_TOOLS](https://github.com/sfarrens/sf_tools)
- [opencv](https://github.com/opencv/opencv-python)
- [tensorflow](https://github.com/tensorflow/tensorflow)
- [sep](https://github.com/kbarbary/sep/tree/v1.1.x)

## Local installation

```bash
git clone https://github.com/CosmoStat/BlendHunter.git
```

## Quick usage

Configuration setup:
```python
from blendhunter.config import BHConfig

bhconfig = BHConfig(config_file='../data/bhconfig.yml').config
in_path = bhconfig['in_path']
out_path = bhconfig['out_path']
noise_sigma = bhconfig['noise_sigma']
n_noise_real = bhconfig['n_noise_real']
sample_range = slice(*bhconfig['sep_sample_range'])
```
Create 35 directories for padded parametric images and prepare data:
```bash
python create_directories.py
python prep_data.py
```
Run BlendHunter network and Source-Extractor as a bechmark:
```bash
python run_bh.py
python run_sep.py
```
### Test on COSMOS images
```python
python test_cosmos.py
```

### Visualize results
Get simulations labels:
```python
labels = np.load(os.path.join(out_path, 'labels.npy')).flatten()
```
Get accuracy on simulations:
```python
from blendhunter.performance import get_acc

bh_mean, bh_std = get_acc('bh_pad_results', out_path, noise_sigma, n_noise_real, labels)
sep_mean, sep_std = get_acc('sep_pad_results', out_path, noise_sigma, n_noise_real, labels)
```

Plot results with error bars:
<div align="center">
  <img src="https://accuracy_plot.png" alt="Accuracy plot">
</div>

Get labels and accuracy on Cosmos images:
```python
labels = np.load(os.path.join(out_path, 'cosmos_labels.npy')).flatten()
bh_mean, bh_std = get_acc('cosmos_results_pad', out_path, noise_sigma, n_noise_real, labels)
```
Plot results:
