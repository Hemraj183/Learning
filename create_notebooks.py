import os
import json

COURSE_MAT = "course_materials"

TEMPLATES = {
    "week4_router": "Building a Router Network",
    "week5_diffusion": "Understanding Diffusion Math",
    "week6_unet": "Implementing U-Net",
    "week7_ldm": "Latent Diffusion Models",
    "week8_capstone": "Capstone: The Noisy Router",
    "week9_lora": "Low-Rank Adaptation (LoRA)",
    "week10_moe": "Mixture of Experts",
    "week11_opt": "Inference Optimization",
    "week12_capstone": "Final Capstone: Ultimate Assistant"
}

def create_skeletons():
    for week, title in TEMPLATES.items():
        folder = os.path.join(COURSE_MAT, week)
        if not os.path.exists(folder): continue
        
        nb_path = os.path.join(folder, "notebook.ipynb")
        if not os.path.exists(nb_path):
            print(f"Creating notebook for {week}...")
            
            content = {
             "cells": [
              {
               "cell_type": "markdown",
               "metadata": {},
               "source": [ f"# {title}\n", "\n", "Welcome to this week's lab!" ]
              },
              {
               "cell_type": "code",
               "execution_count": None,
               "metadata": {},
               "outputs": [],
               "source": [ "import torch\n", "print('Ready to code!')" ]
              }
             ],
             "metadata": {
              "kernelspec": {
               "display_name": "Python 3",
               "language": "python",
               "name": "python3"
              }
             },
             "nbformat": 4,
             "nbformat_minor": 5
            }
            
            with open(nb_path, "w") as f:
                json.dump(content, f, indent=1)

if __name__ == "__main__":
    create_skeletons()
