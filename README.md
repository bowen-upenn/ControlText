## This is the official implementation of the paper [ControlText: Unlocking Controllable Fonts in Multilingual Text Rendering without Font Annotations](https://arxiv.org/abs/2502.10999) in PyTorch.
[![Arxiv](https://img.shields.io/badge/ArXiv-Paper-B31B1B)](https://arxiv.org/abs/2502.10999)

## ✨ Overview

Visual text rendering is a challenging task, especially when precise font control is desired. This work demonstrates that diffusion models can achieve **font-controllable** multilingual text rendering using **just raw images without font label annotations**.

---

## 🚀 Key Takeaways

- **No Font Label Annotations Needed:**  
  A text segmentation model extracts nuanced font details from images, allowing training on large-scale web datasets containing text.
  
- **Ambiguous Font Evaluation:**  
  Introduces quantative metrics `l2@k` and `cos@k` for assessing the quality of fuzzy fonts in the open world without the need of ground-truth font labels.
  
- **User-Driven Design Flexibility:**  
  Adds random perturbations on segmented glyphs, providing flexibility for users who might not align text perfectly when using the front-end.
  
- **Integration with Foundation Models:**  
  Works alongside foundational image generation models for localized text and font editing.

---

## 🔧 How to Train

Our training pipeline uses [PyTorch Lightning](https://www.pytorchlightning.ai/) for scalability and reproducibility. Below is a brief walkthrough:

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


