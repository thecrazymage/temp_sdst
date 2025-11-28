# *CasTex*: Cascaded Text-to-Texture Synthesis via Explicit Texture Maps and Physically-Based Shading

<div align="center">
  <b>WACV 2026</b>
</div>
<br>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2510.17699-b31b1b.svg)](https://arxiv.org/abs/2510.17699)
[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://thecrazymage.github.io/CasTex/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://thecrazymage.github.io/CasTex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>


This repository contains the official implementation of the WACV 2026 paper:
**"CasTex: Cascaded Text-to-Texture Synthesis via Explicit Texture Maps and Physically-Based Shading"**

<!-- Teaser: Используйте GIF или PNG хорошего разрешения -->
<!-- Если есть картинка с коровами, назовите её docs/teaser_cows.png -->
![Teaser image](docs/teaser.png)


### Abstract

This work investigates text-to-texture synthesis using diffusion models to generate **physically-based texture maps** (PBR). We aim to achieve realistic model appearances under varying lighting conditions. A prominent solution for the task is score distillation sampling (SDS), which allows recovering a complex texture using gradient guidance given a differentiable rasterization pipeline.

However, combining SDS with widespread latent diffusion models often produces severe visual artifacts and requires implicit parameterization tricks. As a more direct alternative, we propose **CasTex** — an approach using **cascaded diffusion models**. In our setup, score distillation sampling yields high-quality textures **out-of-the-box**. Specifically, we omit implicit parameterization in favor of **explicit texture maps**, which simplifies the pipeline and improves fidelity. Our experiments show that CasTex significantly outperforms state-of-the-art optimization-based solutions on public benchmarks.

For more details, please check our [**Project Page**](https://thecrazymage.github.io/CasTex/).

---

## Table of Contents
- [Setup Environment](#-installation)
- [Quick Start](#-quick-start)
- [Benchmark on Objaverse subset](#-data-preparation)
- [Citation](#-citation)
---

However, combining SDS with widespread latent diffusion models often produces severe visual artifacts and requires implicit parameterization tricks. As a more direct alternative, we propose **CasTex** — an approach using **cascaded diffusion models**. In our setup, score distillation sampling yields high-quality textures **out-of-the-box**. Specifically, we omit implicit parameterization in favor of **explicit texture maps**, which simplifies the pipeline and improves fidelity. Our experiments show that CasTex significantly outperforms state-of-the-art optimization-based solutions on public benchmarks.However, combining SDS with widespread latent diffusion models often produces severe visual artifacts and requires implicit parameterization tricks. As a more direct alternative, we propose **CasTex** — an approach using **cascaded diffusion models**. In our setup, score distillation sampling yields high-quality textures **out-of-the-box**. Specifically, we omit implicit parameterization in favor of **explicit texture maps**, which simplifies the pipeline and improves fidelity. Our experiments show that CasTex significantly outperforms state-of-the-art optimization-based solutions on public benchmarks.

## Setup Environment



However, combining SDS with widespread latent diffusion models often produces severe visual artifacts and requires implicit parameterization tricks. As a more direct alternative, we propose **CasTex** — an approach using **cascaded diffusion models**. In our setup, score distillation sampling yields high-quality textures **out-of-the-box**. Specifically, we omit implicit parameterization in favor of **explicit texture maps**, which simplifies the pipeline and improves fidelity. Our experiments show that CasTex significantly outperforms state-of-the-art optimization-based solutions on public benchmarks.




## Draft
Credits:
This project includes code adapted from the NVIDIA Kaolin library (Apache 2.0 License).


Пример скрипта:
./scripts/sanity_check.sh -d ./logs/objaverse_eval_26-11-2025_00-00-40/ -ef 1 -ed 2 -esf 6

Parts of the code in `metrics/` (specifically FID calculation) are adapted from [pytorch-fid](https://github.com/mseitzer/pytorch-fid) by Maximilian Seitzer.
