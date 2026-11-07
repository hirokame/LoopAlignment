# Loop Alignment

Code for the NeurIPS 2026 paper

> **Loop Alignment: Self-organized Weight Transpose in Predictive Coding through Independent Hebbian Plasticity**
> Kojiro Hirokane\*, Ryohei Ueno\*, Takashi Kitsukawa (\* equal contribution)

Backpropagation needs the feedback (FB) weights to be an exact transpose of the feedforward (FF) weights. This repository shows that predictive coding (PC) does not need this. Even when the FB pathway is a fixed random matrix that is never updated, the inference loop of PC turns the FF weights toward its transpose, and the network learns as well as one with exact symmetry. We call this Loop Alignment.

All models are written in NumPy. The models use no deep-learning framework, optimizer, nonlinearity or normalization. Keras is used only to download MNIST and Fashion-MNIST.

## Models

All models are in `modules/model.py`. The class names in the code differ from the names in the paper.

| Paper | Class in `modules/model.py` | FB pathway |
|---|---|---|
| Baseline PC | `Whittington_Model` | transpose of the FF weights at every step |
| PC-DH (warm-up) | `PCRHL` | its own matrix, updated by a local Hebbian rule |
| PC-RFB (main model) | `LA_model` | fixed random matrix, never updated |
| Circuit sharing | `dual_LA_model` | two networks; each one's FB passes through the other's FF weights and two relay matrices |

The single-network models run three steps for each training batch.

```python
model.initialize_states()                # reset activities to zero
model.update_x(inputs, target, num_loop) # inference loop, weights fixed
model.update_weight()                    # local Hebbian weight update
```

`model.save_cossim()` returns the cosine similarity between the FF weight from the hidden layer to the output layer and the transpose of the matching FB weight. This is the quantity plotted in the paper.

## Files

| File | What it does | Paper |
|---|---|---|
| `LA_MNISTclass.ipynb` | Trains one model on MNIST and plots accuracy, energy and FF–FB cosine similarity. Choose the model in the training cell. | Fig. 2A–C |
| `LA_loopdiff.ipynb` | Trains PC-RFB with different numbers of inference loops. | Fig. 2E |
| `LA_add_noise.ipynb` | Adds random noise to the weights at every step and compares noise levels. | Fig. 4 |
| `LA_circuitpshare.ipynb` | Two networks share one bidirectional pathway. The relay matrices are fixed. | Fig. 3B–D |
| `DualHebb.ipynb` | Same code as `LA_circuitpshare.ipynb`. | |
| `LA_MNISTclass_Grid_search.ipynb` | Grid search over the two learning rates, shown as a heatmap. The last section reads a CSV file from an earlier search and is not needed. | |
| `modules/model.py` | All models, plus a backpropagation network `BP`. | |
| `modules/func.py` | Activation functions and weight initialization. The paper uses `LINEAR`. | |
| `modules/batch.py` | Turns labels into target vectors, and network outputs back into labels. | |

## Installation

Python 3.9 or newer.

```bash
git clone https://github.com/hirokame/LoopAlignment.git
cd LoopAlignment
pip install -r requirements.txt
```

Keras downloads MNIST and Fashion-MNIST automatically the first time a notebook runs.

`environment.yml` is the full conda environment of one machine we used (Windows). It is kept for reference. `requirements.txt` is enough to run the code on any system.

## Running

Start Jupyter from the repository folder, so that `modules` can be imported, and run a notebook from top to bottom.

```bash
jupyter notebook
```

The settings at the top of each training cell are short, so a first run finishes in a few minutes. Each notebook saves its results as an HDF5 file in its own folder, for example `MNIST_result/`, and then plots them.

## Settings used in the paper

To match the paper, change these variables in the training cell. The learning rates and the other hyperparameters are listed in Appendix A of the paper.

| Paper | Notebook | Change |
|---|---|---|
| Fig. 2A–C | `LA_MNISTclass.ipynb` | `num_step = 1*10**6`, `num_trial = 50`, and the model (`Whittington_Model`, `PCRHL` or `LA_model`) |
| Fig. 2E | `LA_loopdiff.ipynb` | `num_loop_list = [1, 3, 5, 10, 25, 50]`, `num_step = 1*10**6` |
| Fig. 4 | `LA_add_noise.ipynb` | model `LA_model`, noise on the FB matrix only (`add_noise_FB`), `noise_rate = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]`, `num_step = 3*10**6`, `num_trial = 10` |
| Fig. 3B–D | `LA_circuitpshare.ipynb` | `num_step = 3*10**6`, `num_trial = 10`. In the paper Net 2 learns Fashion-MNIST; load it with `fashion_mnist.load_data()` as in `LA_MNISTclass_Grid_search.ipynb` and use it in the PC2 update. |

In our test, one trial of 10<sup>6</sup> training samples took about five minutes on one CPU core.

## Citation

```bibtex
@inproceedings{hirokane2026loop,
  title     = {Loop Alignment: Self-organized Weight Transpose in Predictive Coding through Independent Hebbian Plasticity},
  author    = {Hirokane, Kojiro and Ueno, Ryohei and Kitsukawa, Takashi},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT. See `LICENSE`.
