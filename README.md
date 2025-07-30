
# NVIDIA GenMol (UV-based Setup)

This project provides generative fragment-based molecule design workflows based on NVIDIA's GenMol model. It supports various fragment-to-molecule tasks such as linker design, motif extension, scaffold decoration, and superstructure generation.

https://github.com/NVIDIA-Digital-Bio/genmol

## Installation

This project uses [**uv**](https://docs.astral.sh/uv/getting-started/installation/#installation-methods), a fast Python package manager from Astral. You must install `uv` before proceeding.

### 1. Install `uv` (If you don't have)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository and sync the environment

```bash
git clone https://github.com/XinChenQC/nvidia-genmol-uv.git
cd nvidia-genmol-uv
uv sync
```

This will automatically install all dependencies defined in `pyproject.toml`.

### 3. Download the model checkpoint

```bash
curl -L 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/genmol_v1/1.0/files?redirect=true&path=model.ckpt' -o model.ckpt
```

## Example Usage

```python
from genmol.sampler import Sampler

sampler = Sampler('./model.ckpt')
num_samples = 20

print('Linker Design')
fragment2 = '[11*]N1CCCC1.[17*]c1ccc2c(c1)OCCO2'
samples = sampler.fragment_linking(fragment2, num_samples, randomness=3)
print(samples)

print('Motif Extension')
fragment = '[17*]c1ccc2c(c1)OCCO2'
samples = sampler.fragment_completion(fragment, num_samples, randomness=1.2, gamma=0.3)
print(samples)

print('Scaffold Decoration')
fragment = '[1*]c1cccc(Nc2ncnc3cc([2*])c([3*])cc23)c1'
samples = sampler.fragment_completion(fragment, num_samples, gamma=0.3)
print(samples)

print('Superstructure Generation')
fragment = 'c1ccc(Oc2ccccc2)cc1'
samples = sampler.fragment_completion(fragment, num_samples, gamma=0.4)
print(samples)
```

## Fragment Input Guide

The input fragment SMILES can include **attachment points** indicated by `[*]` or `[1*]`, `[2*]`, etc., which specify the extension sites. For example:

* `[1*]` denotes a specific atom position where the molecule can be extended or modified.
* Multiple asterisks like `[1*][2*][3*]` allow defining multiple connection points for scaffold decoration.

## Supported Tasks

* **Linker Design / Scaffold Morphing** – Generate a linker connecting two side chains
* **Motif Extension** – Grow a molecule from a given motif fragment
* **Scaffold Decoration** – Add decorations to a larger scaffold
* **Superstructure Generation** – Generate a full molecule given a partial structure
* **Linker Design (1-step)** – Link two fragments without intermediate mixing


# Try it on ChemOrchestra

You can also try out our integrated product for free directly on the [ChemOrchestra](https://www.quantabricks.xyz) platform.

![0730-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/8a1915cd-c743-4bf7-9729-0a34a67515ee)

ChemOrchestra is an all-in-one platform designed to accelerate drug discovery through an integrated suite of user-friendly computational tools. It allows researchers to quickly build and execute complex drug development workflows—without the need for extensive coding or setup. Tools like MolGen, featured prominently on the main page, make molecular generation and screening accessible with just a few clicks. Whether you're designing ligands, running docking studies, ChemOrchestra streamlines the process for both experts and newcomers in computational drug design.

You can find GenMol on the Mainpage

<img width="1187" height="674" alt="be2ff1a7a1e2e44a17b203d9cdc7d075" src="https://github.com/user-attachments/assets/181166a1-b874-4e7a-b9a7-d5d3be530237" />


## Wrapper License

This wrapper (outside genmol), interacts with NVIDIA code, is licensed under the MIT License.

## NVIDIA-part License
Copyright @ 2025, NVIDIA Corporation. All rights reserved.<br>
The source code is made available under Apache-2.0.<br>
The model weights are made available under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## Citation
If you find this repository and NVIDIA‘s paper useful, you must cite NVIDIA's work.
```BibTex
@article{lee2025genmol,
  title     = {GenMol: A Drug Discovery Generalist with Discrete Diffusion},
  author    = {Lee, Seul and Kreis, Karsten and Veccham, Srimukh Prasad and Liu, Meng and Reidenbach, Danny and Peng, Yuxing and Paliwal, Saee and Nie, Weili and Vahdat, Arash},
  journal   = {International Conference on Machine Learning},
  year      = {2025}
}
```

