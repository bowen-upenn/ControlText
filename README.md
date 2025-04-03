## This is the official implementation of the paper [ControlText: Unlocking Controllable Fonts in Multilingual Text Rendering without Font Annotations](https://arxiv.org/abs/2502.10999) in PyTorch.
[![Arxiv](https://img.shields.io/badge/ArXiv-Paper-B31B1B)](https://arxiv.org/abs/2502.10999)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Cite_Our_Paper-4085F4)](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C39&q=ControlText%3A+Unlocking+Controllable+Fonts+in+Multilingual+Text+Rendering+without+Font+Annotations&btnG=)

## ✨ Overview

Visual text rendering is a challenging task, especially when precise font control is desired. This work demonstrates that diffusion models can achieve **font-controllable** multilingual text rendering using **just raw images without font label annotations**.

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

![Banner](banner.png)
---

## Citation
If you find our work inspires you, please consider citing it. Thank you!

    @article{jiang2025controltext,
      title={ControlText: Unlocking Controllable Fonts in Multilingual Text Rendering without Font Annotations},
      author={Jiang, Bowen and Yuan, Yuan and Bai, Xinyi and Hao, Zhuoqun and Yin, Alyson and Hu, Yaojie and Liao, Wenyu and Ungar, Lyle and Taylor, Camillo J},
      journal={arXiv preprint arXiv:2502.10999},
      year={2025}
    }
    
## 🔧 How to Train

Our repository is based on the code of [AnyText](https://github.com/tyxsspa/AnyText). We build upon and extend it to enable user-controllable fonts in zero-shot. Below is a brief walkthrough:

1. **Prerequisites:**
   We use conda environment to manage all required packages.
    ```
    conda env create -f environment.yml
    conda activate controltext
    ```

3. **Preprocess Glyphs:**


5. **Configuration:**
   - Adjust hyperparameters such as `batch_size`, `grad_accum`, `learning_rate`, `logger_freq`, and `max_epochs` in the training script `train.py`. Please keep `mask_ratio = 1`.
   - Set paths for GPUs, checkpoints, model configuration file, image datasets, and preprocessed glyphs accordingly.

6. **Training Command:**
   Run the training script:
   ```bash
   python train.py
   ```
   
## 🔮 Inference & Front-End

The front-end code for user-friendly text and font editing are coming soon! Stay tuned for updates as we continue to enhance the project.

## 👩‍💻 Evaluation
1. **Our Generated Data**
   
   laion_controltext [Google Drive](https://drive.google.com/file/d/1sxzAENTWDAixkMFMHyOeXcyhZOq7WY2B/view?usp=sharing), laion_controltext_gly_lines (cropped regions for each line of text from the entire image) [Google Drive](https://drive.google.com/file/d/1JrJTkJ8oePXUo9d8E5QOVsh0DBWi82P_/view?usp=sharing), laion_controltext_gly_lines_grayscale (laion_controltext_gly_lines after text segmentation) [Google Drive](https://drive.google.com/file/d/1qSQs_NB3jUe08YZLaKmM42iJWqT7mmjA/view?usp=drive_link), laion_gly_lines_gt (cropped regions from input glyphs after text segmentation) [Google Drive](https://drive.google.com/file/d/1XiRu24gRiYwpODyjuJnW1XyJ9kd-1f9U/view?usp=drive_link)

 
   wukong_controltext [Google Drive](https://drive.google.com/file/d/1ZCeEsD4aCeK0OePNUHQ96Pp3Xq_f4pW2/view?usp=drive_link), wukong_controltext_gly_line [Google Drive](https://drive.google.com/file/d/1weseRPN5mNA2NNeOjUxFQxA7Fu6K4CuZ/view?usp=drive_link), wukong_controltext_glylines_grayscale [Google Drive](https://drive.google.com/file/d/1uyWyF_FwMhyAyVRsBQsTx9G7Ar5dZBMb/view?usp=drive_link), wukong_gly_lines_gt [Google Drive](https://drive.google.com/file/d/1XKsliU0-XVxj7YUyfbGaAq1PCED18s1a/view?usp=drive_link)

3. **Our Model Checkpoint**

     [Google Drive](https://drive.google.com/file/d/1fUNeKqoGhGutkcCFTHa3USkhChlfE_kQ/view?usp=sharing)

5. **Script for evaluating text accuracy:**

      Run the following script to calculate SenACC and NED scores for text accuracy, which will evaluate ```laion_controltext_gly_lines``` and ```wukong_controltext_gly_line```.
      ```
      bash eval/eval_dgocr.sh
      ```
      Run the following script to calculate FID score for overall image quality, which will evaluate ```laion_controltext``` and ```wukong_controltext```.
      ```
      bash eval/eval_fid.sh
      ```

7. **Script for evaluating font accuracy in the open world:**

     Run the following script to calculate the font accuracy
     ```
     bash eval/eval_font.sh --generated_folder path/to/your/generated_folder --gt_folder path/to/your/gt_folder
     ```
     In the argument, ```path/to/your/generated_folder``` should point to the directory containing your generated images, for example, ```laion_controltext_gly_lines_grayscale``` or ```wukong_controltext_glylines_grayscale```. Similarly, ```path/to/your/gt_folder``` should refer to the directory containing the ground-truth glyph images or the segmented glyphs used as input conditions, where we use ```laion_gly_lines_gt``` or ```wukong_gly_lines_gt```.

---
![Flows](flows.png)
