# *CasTex*: Cascaded Text-to-Texture Synthesis via Explicit Texture Maps and Physically-Based Shading
[![arXiv](https://img.shields.io/badge/arXiv-2510.17699-b31b1b.svg)](https://arxiv.org/abs/2510.17699)
[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://thecrazymage.github.io/CasTex/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://thecrazymage.github.io/CasTex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div align="center">
  <h3><span style="color:green">🎉 Accepted to WACV 2026 🎉</span></h3>
</div>

<br>

This repository contains the official implementation of the WACV 2026 paper:
<br>
**CasTex: Cascaded Text-to-Texture Synthesis via Explicit Texture Maps and Physically-Based Shading** 
<br>
by [Aliev Mishan](https://scholar.google.com/citations?user=QJz42PEAAAAJ&hl=en), [Dmitry Baranchuk](https://scholar.google.com/citations?user=NiPmk8oAAAAJ&hl=en&oi=ao), [Kirill Struminsky](https://scholar.google.com/citations?hl=en&user=q69zIO0AAAAJ).

<!-- Teaser: Используйте GIF или PNG хорошего разрешения -->
<!-- Если есть картинка с коровами, назовите её docs/teaser_cows.png -->
<!-- ![Teaser image](docs/teaser.png) -->


### Abstract

This work investigates text-to-texture synthesis using diffusion models to generate physically-based texture maps.
We aim to achieve realistic model appearances under varying lighting conditions.
A prominent solution for the task is score distillation sampling.
It allows recovering a complex texture using gradient guidance given a differentiable rasterization and shading pipeline.
However, in practice, the aforementioned solution in conjunction with the widespread latent diffusion models produces severe visual artifacts and requires additional regularization such as implicit texture parameterization.
As a more direct alternative, we propose an approach using cascaded diffusion models for texture synthesis (CasTex).
In our setup, score distillation sampling yields high-quality textures out-of-the box.
In particular, we were able to omit implicit texture parameterization in favor of an explicit parameterization to improve the procedure.
In the experiments, we show that our approach significantly outperforms state-of-the-art optimization-based solutions on public texture synthesis benchmarks.

For more details, please check our [Project Page](https://thecrazymage.github.io/CasTex/).

If you are interested in collaborating, please reach out to us via [alievmishan78@gmail.com](mailto:alievmishan78@gmail.com).


## Table of Contents
- [Setup Environment](#setup-environment)
- [Quick Start](#quick-start)
- [Benchmark on Objaverse subset](#benchmark-on-objaverse-subset)
- [Acknowledgement](#???)
- [Citation](#citation)
- [License](#license)

## Setup Environment

See [environment.yml](environment.yml) for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:

```.bash
./bash/setup_environment.sh
```

## Quick Start

To make sure everything is set up and configured correctly, you can run the following script to generate texture for the Spot.

```.bash
./bash/run.sh
```

## Benchmark on Objaverse subset

To generate textures for the Objaverse objects as was proposed in Text2Tex paper, you need to run this commands from the root of your directory:
1) download Blender 3.3.21
```.bash
./bash/download_blender.sh
```

2) download and pre-process Objaverse objects
```.bash
./bash/objaverse_data_processing.sh
```

3) render ground truth objects
```.bash
pupupuou
```

4) genearte texture for preprocessed objects
```.bash
pupupuou
```

5) genearte textures for preprocessed objects
```.bash
./bash/run_objaverse_eval.sh
```

6) render generated textures
```.bash
pupupuou
```

7) calculate FID/KID
```.bash
pupupuou
```

## Acknowledgement

## Citation

```bibtex
@article{aliev2025castex,
  title={CasTex: Cascaded Text-to-Texture Synthesis via Explicit Texture Maps and Physically-Based Shading},
  author={Aliev, Mishan and Baranchuk, Dmitry and Struminsky, Kirill},
  journal={arXiv preprint arXiv:2504.06856},
  year={2025}
}
```


## License

Hello, darling)
<!-- ## Draft
Credits:
This project includes code adapted from the NVIDIA Kaolin library (Apache 2.0 License).


Пример скрипта:
./scripts/sanity_check.sh -d ./logs/objaverse_eval_26-11-2025_00-00-40/ -ef 1 -ed 2 -esf 6

./scripts/render_frames.sh ./logs/test/ ii test

./scripts/run_metrics.sh ./objaverse_eval/renders/ours_2025_07_15_XL_L/frames/

Parts of the code in `metrics/` (specifically FID calculation) are adapted from [pytorch-fid](https://github.com/mseitzer/pytorch-fid) by Maximilian Seitzer. -->