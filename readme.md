

# Gradio Demo for syncSDE with Conda

Official implementation of the paper

```
   @inproceedings{
      anonymous2026semantic,
      title={Semantic Editing with Coupled Stochastic Differential Equations},
      author={Anonymous},
      booktitle={Forty-third International Conference on Machine Learning},
      year={2026},
      url={https://openreview.net/forum?id=kaOPDyq5sv}
   }
```

## Setup

1. **Create and activate a Conda environment:**

   ```bash
   conda create -n syncSDE python=3.13 -y
   conda activate syncSDE
   ```

2. **Install the essential runtime packages:**

   ```bash
   conda install -c pytorch -c nvidia -c conda-forge \
      pytorch torchvision pytorch-cuda=12.8 \
      numpy pillow matplotlib gradio accelerate transformers \
      sentencepiece protobuf safetensors einops pip -y

   python -m pip install "diffusers>=0.38.0" "bitsandbytes>=0.49.2"
   ```

   `diffusers>=0.38.0` is needed for FLUX.2 support, and `bitsandbytes` is needed for the default 4-bit FLUX.2 demo path.

## Run Demo

1. **Start the Gradio demo:**

   ```bash
   python gradio_demo_flux.py --share
   ```

2. **Open the web UI:**

   * A public link will appear in the terminal (starting with `https://...`).
   * Open the link in your browser and interact with the demo.

## Credits
Our code is built upon the code release https://github.com/LituRout/RF-Inversion of the paper

```
    @inproceedings{
        rout2025semantic,
        title={Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations},
        author={Litu Rout and Yujia Chen and Nataniel Ruiz and Constantine Caramanis and Sanjay Shakkottai and Wen-Sheng Chu},
        booktitle={The Thirteenth International Conference on Learning Representations},
        year={2025},
        url={https://openreview.net/forum?id=Hu0FSOSEyS}
    }
```
