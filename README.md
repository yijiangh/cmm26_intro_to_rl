# CMM 2026 RL Lecture Demos

This directory contains the lecture notebooks, figure-generation scripts, and crawler demos used for the 2026 "Computational Models of Motion" RL lectures.

The main runnable pieces are:

- `L6-0_demo_gridworld_dp.ipynb`: exact methods on gridworld
- `L6-1_demo_crawler_q-learning.ipynb`: value-based methods on the 2D MuJoCo crawler
- `L6-2_demo_crawler_pg.ipynb`: policy-gradient methods on the same crawler
- `teleop_crawler.py`: interactive crawler teleoperation demo
- `scripts/*.py`: one-off figure generators for lecture visuals

## 1. Prerequisites

Recommended local setup:

- Python 3.10 or 3.11
- Ubuntu or macOS
- a working GUI environment if you want to run `teleop_crawler.py`

Notes:

- `mujoco` is installed from PyPI; no separate MuJoCo download is required for this repo.
- `ffmpeg` is required if you want notebook cells that export animations or videos to work. Without it, Matplotlib video export fails with `RuntimeError: Requested MovieWriter (ffmpeg) not available`.
- `tkinter` is required for `teleop_crawler.py`. It is usually available on macOS system Python installations. If your Python build does not include it, the teleop script will not launch.
- `uv` works on macOS, Linux, and Windows, but this repo has only been tested locally on Ubuntu and macOS.
- This repo now includes a minimal `pyproject.toml` and `.python-version`, so `uv sync` can create and populate the environment directly from project metadata.
- On Linux, `pip install torch` may install a CUDA-enabled wheel. If you specifically want a CPU-only PyTorch build, use the selector on the official PyTorch install page instead of the default command below.

Install `ffmpeg` before running notebooks that save videos:

Ubuntu:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

macOS:

```bash
brew install ffmpeg
```

## 2. Create and configure the environment

Recommended `uv` workflow from this directory:

```bash
python3 -m pip install --user uv
~/.local/bin/uv sync
```

If `uv` is on your `PATH`, you can replace `~/.local/bin/uv` with `uv`.

`uv sync` will create `.venv/`, resolve from `pyproject.toml`, and install the pinned dependencies into that environment.

If you prefer not to use `uv`, the `virtualenv` fallback also works:

```bash
python3 -m pip install --user virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  numpy \
  matplotlib \
  pillow \
  ipython \
  ipykernel \
  notebook \
  jupyterlab \
  ipywidgets \
  mujoco \
  gymnasium \
  torch \
  stable-baselines3
```

If your system already has `python3-venv` installed and you prefer the stdlib tool, this equivalent alternative also works:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  numpy \
  matplotlib \
  pillow \
  ipython \
  ipykernel \
  notebook \
  jupyterlab \
  ipywidgets \
  mujoco \
  gymnasium \
  torch \
  stable-baselines3
```

Verified locally in this repo on March 26, 2026 with Python `3.10.12`:

- `numpy 2.2.6`
- `matplotlib 3.10.8`
- `pillow 12.1.1`
- `ipython 8.38.0`
- `ipykernel 7.2.0`
- `notebook 7.5.5`
- `jupyterlab 4.5.6`
- `ipywidgets 8.1.8`
- `mujoco 3.6.0`
- `gymnasium 1.2.3`
- `torch 2.11.0+cu130`
- `stable-baselines3 2.7.1`

Register the environment as a Jupyter kernel:

```bash
.venv/bin/python -m ipykernel install --user --name rl-lectures --display-name "Python (rl-lectures)"
```

With `uv`, you can also run notebook commands without manually activating first:

```bash
~/.local/bin/uv run jupyter lab
```

Quick verification:

```bash
.venv/bin/python -c "import mujoco, torch, gymnasium, stable_baselines3; print(mujoco.__version__, torch.__version__)"
```

## 3. Activate the environment in later sessions

Every time you come back to the repo:

```bash
cd /path/to/rl_lectures
source .venv/bin/activate
```

## 4. Run the notebooks

Start Jupyter:

```bash
jupyter notebook
```

Open the notebooks in this order:

1. `L6-0_demo_gridworld_dp.ipynb`
2. `L6-1_demo_crawler_q-learning.ipynb`
3. `L6-2_demo_crawler_pg.ipynb`

Notebook notes:

- In each notebook, run cells from top to bottom.
- The crawler notebooks include `!pip install -q mujoco` in their setup cells for Colab compatibility. Locally, if your `.venv` is already configured, that cell is harmless but usually unnecessary.
- The crawler notebooks cache trained models in `saved_checkpoints/` and may reuse them automatically.
- If you want to force retraining, set `FORCE_RETRAIN = True` in the relevant notebook helper cell.
- `L6-2_demo_crawler_pg.ipynb` expects the Lecture 1 notebook artifacts to exist if you want to compare against saved Lecture 1 policies from `saved_policies/`.

## 5. Run the interactive crawler teleop demo

Continuous torque control:

```bash
.venv/bin/python teleop_crawler.py --mode continuous
```

Discrete 4-action control:

```bash
.venv/bin/python teleop_crawler.py --mode discrete
```

If you already activated the environment, the equivalent commands are:

```bash
python teleop_crawler.py --mode continuous
python teleop_crawler.py --mode discrete
```

## 6. Run the figure-generation scripts

These scripts regenerate lecture figures under `generated_figures/`.

Gridworld intro figure:

```bash
python scripts/generate_gridworld_intro_visual.py
```

Likelihood-ratio policy-gradient figures:

```bash
python scripts/generate_likelihood_ratio_visuals.py
```

Variance-reduction figures:

```bash
python scripts/generate_variance_reduction_visuals.py
```

## 7. Outputs and cached artifacts

Running the notebooks and scripts will read from or write to these directories:

- `generated_figures/`: regenerated static figures for slides and notebooks
- `saved_checkpoints/`: notebook training checkpoints
- `saved_policies/`: exported reusable policies
- `saved_rollouts/`: saved rollout videos/data

These folders are part of the workflow. Do not delete them unless you intentionally want to discard cached outputs.

## 8. Colab usage

The notebooks are written to be Colab-friendly:

- the setup cells install `mujoco`
- the crawler environments are defined inline in the notebooks
- no separate package install step from this README is required in Colab beyond running the notebook cells

For local work, use the `.venv` instructions above instead of relying on per-notebook installs.

## 9. Troubleshooting

`ModuleNotFoundError`

- Make sure the `.venv` is activated.
- Make sure the notebook is using the `Python (rl-lectures)` kernel.

`teleop_crawler.py` opens no window or crashes on startup

- Check that `tkinter` is available in your Python build.
- Run the teleop script from a normal desktop session, not a headless shell.

`RuntimeError: Requested MovieWriter (ffmpeg) not available`

- Install the system `ffmpeg` package.
- On Ubuntu: `sudo apt install -y ffmpeg`
- On macOS: `brew install ffmpeg`

Matplotlib cache or font-cache warnings

- In restricted environments, set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

Notebook retrains instead of loading a checkpoint

- Check that `saved_checkpoints/` exists in this directory.
- Make sure you launched Jupyter from this repo root.
- Check whether `FORCE_RETRAIN` was set to `True`.

## 10. Minimal local workflow

If you just want the shortest path from clone to running notebooks:

```bash
cd /path/to/rl_lectures
python3 -m pip install --user uv
~/.local/bin/uv sync
.venv/bin/python -m ipykernel install --user --name rl-lectures --display-name "Python (rl-lectures)"
~/.local/bin/uv run jupyter lab
```
