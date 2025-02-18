## This is the official implementation of the paper [ControlText: Unlocking Controllable Fonts in Multilingual Text Rendering without Font Annotations](https://arxiv.org/abs/2502.10999) in PyTorch.
[![Arxiv](https://img.shields.io/badge/ArXiv-Paper-B31B1B)](https://arxiv.org/abs/2502.10999)

## ✨ Overview

Visual text rendering is a challenging task, especially when precise font control is desired. This work demonstrates that diffusion models can achieve **font-controllable** multilingual text rendering using **just raw images without font label annotations**.

![Banner](banner.png)

---

## 🚀 Key Takeaways

- **Font controls require no font label annotations:**  
  A text segmentation model can capture nuanced font information in pixel space without requiring font label annotations in the dataset, enabling zero- shot generation on unseen languages and fonts, as well as scalable training on web-scale image datasets as long as they contain text.
  
- **Evaluating ambiguous fonts in the open world:**
  Fuzzy font accuracy can be measured in the embed- ding space of a pretrained font classification model, utilizing our proposed metrics `l2@k` and `cos@k`.
  
- **Supporting user-driven design flexibility:**  
  Random perturbations can be applied to segmented glyphs. While this won’t affect the rendered text quality, it accounts for users not precisely aligning text to best locations and prevents models from rigidly replicating the pixel locations in glyphs.
  
- **Working with foundation models:**  
  With limited computational resources, we can still copilot foundational image generation models to perform localized text and font editing.

---

## 🔧 How to Train

Our repository is based on the code of [AnyText](https://github.com/tyxsspa/AnyText). We build upon and extend it to enable user-controllable fonts in zero-shot. Below is a brief walkthrough:

1. **Prerequisites:**
   We use conda environment to manage all required packages.
    ```
    conda env create -f environment.yaml
    conda activate controltext
    ```

3. **Preprocess Glyphs:**

4. **Configuration:**
   - Adjust hyperparameters such as `batch_size`, `grad_accum`, `learning_rate`, `logger_freq`, and `max_epochs` in the training script `train.py`. Please keep `mask_ratio = 1`.
   - Set paths for GPUs, checkpoints, model configuration file, image datasets, and preprocessed glyphs accordingly.

5. **Training Command:**
   Run the training script:
   ```bash
   python train.py
   ```
   
## 🔮 Inference & Front-End

The front-end code for user-friendly text and font editing are coming soon! Stay tuned for updates as we continue to enhance the project.


