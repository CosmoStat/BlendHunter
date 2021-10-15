![blendhunter](https://user-images.githubusercontent.com/7417573/127934298-39734525-6325-4d98-900d-136227f03b38.png)

Deep learning tool for identifying blended galaxy images in survey images.

---
> Main contributors:  
> - <a href="https://github.com/sfarrens" target="_blank" style="text-decoration:none; color: #F08080">Samuel Farrens</a>  
> - <a href="https://github.com/ablacan" target="_blank" style="text-decoration:none; color: #F08080">Alice Lacan</a>
> - <a href="https://github.com/aguinot" target="_blank" style="text-decoration:none; color: #F08080">Axel Guinot</a>
> - <a href="https://github.com/andrevitorelli" target="_blank" style="text-decoration:none; color: #F08080">André Zamorano Vitorelli</a>
---

BlendHunter deep transfer learning based approach for the automated and robust identification of blended sources in galaxy survey data.

## Dependencies
The following python packages should be installed with their specific dependencies:

- [Numpy](https://github.com/numpy/numpy)
- [ModOpt](https://github.com/CEA-COSMIC/ModOpt)
- [LMFIT](https://lmfit.github.io/lmfit-py/)
- [SF_Tools](https://github.com/sfarrens/sf_tools)
- [OpenCV](https://github.com/opencv/opencv-python)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [SEP](https://github.com/kbarbary/sep/tree/v1.1.x)

## Local installation

```bash
$ git clone https://github.com/CosmoStat/BlendHunter.git
$ pip install -e .
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

### notebooks
Find the jupyter notebooks for results visualization.

### sextractor
Find the scripts to run SExtractor.
