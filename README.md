# *CasTex*: Cascaded Text-to-Texture Synthesis via Explicit Texture Maps and Physically-Based Shading
[![arXiv](https://img.shields.io/badge/arXiv-2510.17699-b31b1b.svg)](https://arxiv.org/abs/2510.17699)
[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://thecrazymage.github.io/CasTex/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://thecrazymage.github.io/CasTex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div align="center">
  <img src="https://img.shields.io/badge/🎉_Accepted_to_WACV_2026_🎉-2ea44f?style=for-the-badge&labelColor=2ea44f" alt="Accepted to WACV 2026"/>
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

For more details and results, please visit our [Project Page](https://thecrazymage.github.io/CasTex/).

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmark on Objaverse subset](#benchmark-on-objaverse-subset)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

## Installation

Please refer to [environment.yml](environment.yml) for the complete list of dependencies. To create and activate the environment using Miniconda3, run the following commands:

```.bash
git clone https://github.com/thecrazymage/CasTex.git
cd CasTex

bash scripts/setup_environment.sh
conda activate castex
```

## Quick Start

To generate a texture for a single object using a text prompt:
```.bash
bash scripts/run.sh
```
The generated PBR textures for stages `i` and `ii` will be saved in `logs/`.

## Benchmark on Objaverse subset

To generate textures for the Objaverse objects using the protocol from the [Text2Tex](https://arxiv.org/abs/2303.11396) paper, run the following command from the root directory:

1) Download Blender 3.3.21:
    ```.bash
    bash scripts/download_blender.sh
    ```
    This will download and extract **Blender** to `objaverse_eval/blender-3.3.21-linux-x64`.

2) To download and process the Objaverse subset:
    ```.bash
    bash scripts/objaverse_data_processing.sh
    ```
    This script downloads the original `.glb` models from the Objaverse dataset and converts them into clean `.obj` files ready for texturing.

    **Output locations:**
    - Original files: `objaverse_eval/objaverse_data/glbs`
    - Processed meshes: `objaverse_eval/objaverse_data/obj`

3) To generate the reference images for metric (FID/KID) calculation:
    ```.bash
    bash scripts/render_gt_frames.sh
    ```
    This script renders the original (ground truth) `.glb` files from the Objaverse subset using Blender.
    
    **Output:** All ground truth renders will be saved in `objaverse_eval/renders/ground_truth/`.

    *These renders are required to compute FID and KID scores.*

4) To generate textures for the preprocessed objects:
    ```.bash
    bash scripts/run_objaverse_eval.sh
    ```
    This script launches the generation process. It is optimized for multi-GPU setups (e.g., SLURM clusters).

    **Output:** Generated textures and logs will be saved in `logs/objaverse_eval_{date}`.

    > **Verification:** You can quickly check if the generation finished correctly by running the sanity check script on the output folder:
    > ```
    > bash scripts/sanity_check.sh -d logs/objaverse_eval_{date} -ef 1 -ed 2 -esf 6
    > ```

6) render generated textures
    ```.bash
    scripts/render_frames.sh logs/objaverse_eval_{date}/ ii objaverse_eval_{date}
    ```
    After training you need to render your trained textures from stage `ii` with Blender. All rendered data will be in `objaverse_eval/renders/objaverse_eval_{date}/`. This command also render videos for future side-by-side comparision.

7) calculate FID/KID
    ```.bash
    scripts/run_metrics.sh objaverse_eval/renders/objaverse_eval_{date}/frames/
    ```
    After all preparations we finally can calculate FID/KID metrics. 

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