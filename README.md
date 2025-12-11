# Tree-based Model Approximation

This repository contains code and notebooks for approximating, analyzing, and reconstructing decision trees using soft (differentiable) surrogates and counterfactual-based active querying (FOCUS). The main notebook is `approx_tree.ipynb` which includes:

- Building soft (differentiable) versions of sklearn decision trees
- Parametric soft-tree surrogate models implemented in TensorFlow
- Counterfactual search (single and batched) using Adam optimization
- FOCUS query generation and surrogate training
- Reconstruction experiments (Iris, MNIST subset) and fidelity plots

## Getting started

Prerequisites
- Python 3.8+ (the code was developed on Python 3.12)
- A virtual environment is strongly recommended

Install dependencies (example):

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, the core packages used are:

- numpy
- scikit-learn
- matplotlib
- tensorflow

## Running the notebook

Open `approx_tree.ipynb` in Jupyter or VS Code and run the cells top to bottom. The notebook includes small smoke-test experiments for Iris and a subset of MNIST. Use smaller budgets/settings for quick tests.

Notes
- The notebook uses a cached batched counterfactual solver to avoid creating `tf.Variable` inside a `@tf.function` repeatedly.
- TensorFlow may emit retracing warnings when batch shapes vary; these are performance warnings (not correctness errors).

## Structure

- `approx_tree.ipynb` — main notebook with all code and experiments

## License

This repository is provided for research and educational purposes. Update the license as needed.
