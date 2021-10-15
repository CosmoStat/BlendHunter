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

## Reproducible Research

To repeat experiments carried out in [Farrens et al. (2021)](...) or to carry out similar experiments on a different data set you will need to go through the following steps.

### Download Data

You can download the parametric training data and realistic CFIS-like images [here]().

Alternatively, you can use your own data provided it is formatted in the same way.

### Configuration Setup

You will need to modify the `bhconfig.yml` file in the `data` directory. Specifically, specifying the paths to the input data and where outputs should be written.

The structure of this file is as follows

```yml
out_path: ...
in_path: ...
cosmos_path: ...
noise_sigma:
  - 5
  - 10
  - 15
  - 20
  - 25
  - 30
  - 35
n_noise_real: 10
sep_sample_range:
  - 36000
  - 40000
cosmos_sample_range:
  - 0
  - 10000
```

where:

- `out_path` specifies the path where outputs should be written.
- `in_path` specifies the path to the input parametric model training data.
- `cosmos_path` specifies the path to the input realistic testing data.
- `noise_sigma` specifies the list of noise standard deviations that should be added to the training data.
- `n_noise_real` specifies the number of noise realisations that should be made for each noise level.
- `sep_sample_range` specifies the range of objects in the training sample on which SEP should be run.
- `cosmos_sample_range` specifies the range of objects...

### Prepare Data

Once you have downloaded (or formatted) your data and updated the configuration file you should run the following scrips in the `scripts` directory.

```bash
$ python scripts/create_directories.py
```

This will prepare directories in your output path to store all of the output products.

```bash
$ python scripts/prep_data.py
```

This will prepare the training and testing data set by padding, adding noise and converting to PNG files.

> :warning: Each noise level and realisation will constitute an independent data set and increase the amount of storage space required. *e.g.* 7 noise levels and 10 realisations will constitute 70 times the volume of data.

### Run BlendHunter and SEP

Run BlendHunter to train the network on the parametric training data. This additionally tests the resulting weights by making predictions on a subsample of this data reserved for testing.

```bash
python run_bh.py
```

Run SEP on the subsample of testing data for comparison.

```
python run_sep.py
```

### Test on CFIS-like images

Run both BlendHunter and SEP on the realistic CFIS-like testing data.

```python
python test_cosmos.py
```

### Notebooks

Finally, in the `notebooks` directory, you will find several Jupyter notebooks where you can reproduce the plots in the paper or make equivalent plots for a different data set.
