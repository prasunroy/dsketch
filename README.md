### Official code for d-Sketch: Improving Visual Fidelity of Sketch-to-Image Translation with Pretrained Latent Diffusion Models without Retraining.

*Accepted in the International Conference on Pattern Recognition (ICPR) 2024.*

[![badge_torch](https://img.shields.io/badge/made_with-PyTorch_2.0-EE4C2C?style=flat-square&logo=PyTorch)](https://pytorch.org/)
[![badge_arxiv](https://img.shields.io/badge/arXiv-2502.14007-brightgreen?style=flat-square)](https://arxiv.org/abs/2502.14007)

![teaser](https://github.com/user-attachments/assets/42cc1fe5-7c2d-4b66-b4e2-932674357426)

### :zap: Getting Started
> Note: This release is tested on Python 3.9.16.
```bash
git clone https://github.com/prasunroy/dsketch.git
cd dsketch
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### :fire: Training
* Download the [Flickr20 dataset](https://drive.google.com/file/d/1ORprRYC2wz3RqI2nyN6YLOA8YmEUhJWK/view) and extract into `datasets/flickr20` directory.
* Run `lctn_train.py` with the following options.
```bash
lctn_train.py [-h] [--sd_path SD_PATH] [--mixed_precision {no,fp16,bf16,fp8}] [--force_cpu]
              [--data_root DATA_ROOT] [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE]
              [--shuffle] [--num_workers NUM_WORKERS] [--lr LR] [--steps STEPS]
              [--output_freq OUTPUT_FREQ] [--output_root OUTPUT_ROOT]
```
##### Example
```bash
python lctn_train.py --sd_path stabilityai/stable-diffusion-2-1 --mixed_precision fp16 --data_root ./datasets/flickr20/ --image_size 768 --batch_size 4 --shuffle --num_workers 8 --lr 0.001 --steps 50000 --output_freq 100 --output_root ./output/
```

### :sparkles: Sampling
* Download the [sample sketches](https://drive.google.com/file/d/1NvTniC2N9GQ4P3R6rxuVxTQu0B0Sp9uW/view) and extract into `result/sample_sketches` directory.
* (Optional) Copy the best checkpoint `<OUTPUT_ROOT>/<TIMESTAMP>/lctn.pth` into `checkpoints` directory.
* Run `lctn_sample.py` with the following options.
```bash
lctn_sample.py [-h] [--seed SEED] [--prompt PROMPT] [--sketch SKETCH] [--image_size IMAGE_SIZE]
               [--guidance_scale GUIDANCE_SCALE] [--noising_scale NOISING_SCALE] [--steps STEPS]
               [--sd_path SD_PATH] [--lctn_path LCTN_PATH] [--mixed_precision {no,fp16,bf16,fp8}]
               [--force_cpu] [--output_dir OUTPUT_DIR]
```
##### Example
```bash
python lctn_sample.py --seed 11111111 --prompt "photo of a fox" --sketch ./result/sample_sketches/fox.png --image_size 768 --guidance_scale 8.0 --noising_scale 0.8 --steps 50 --sd_path stabilityai/stable-diffusion-2-1 --lctn_path ./checkpoints/lctn_flickr20.pth --mixed_precision fp16 --output_dir ./result/fox/
```

### :heart: Citation
```
@inproceedings{roy2022dsketch,
  title     = {d-Sketch: Improving Visual Fidelity of Sketch-to-Image Translation with Pretrained Latent Diffusion Models without Retraining},
  author    = {Roy, Prasun and Bhattacharya, Saumik and Ghosh, Subhankar and Pal, Umapada and Blumenstein, Michael},
  booktitle = {The International Conference on Pattern Recognition (ICPR)},
  month     = {December},
  year      = {2024}
}
```

### :page_facing_up: License
```
Copyright 2024 by the authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

##### Made with :heart: and :pizza: on Earth.
