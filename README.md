# Fine-Tuning Qwen3-14B for Biomedical Question Answering

This repository contains a Jupyter notebook demonstrating how to efficiently fine-tune the `Qwen3-14B` model for a biomedical question-answering task using the `pubmed_qa` dataset.

The project leverages the **Unsloth** library to achieve faster training speeds and less memory usage, making it possible to fine-tune a 14-billion-parameter model on a single free-tier Google Colab GPU (T4).

## Project Overview

The goal of this project is to create a specialized language model that can answer biomedical questions based on a given scientific context. The model is trained to provide a direct `yes/no/maybe` answer followed by a detailed explanation, mimicking how a human expert would respond.

### Key Features

-   **Model:** `Qwen3-14B`, a powerful open-source model.
-   -   **Dataset:** [`pubmed_qa`](https://huggingface.co/datasets/pubmed_qa), pqa_artificial split
-   **Technique:** Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** (Low-Rank Adaptation).
-   **Acceleration:** Optimized with **Unsloth** for high-performance training and 4-bit quantization.
-   **Outcome:** A fine-tuned LoRA adapter that can be loaded on top of the base Qwen3-14B model to perform specialized biomedical Q&A.

## How It Works

The `Qwen3_(14B)_PubMed_QA.ipynb` notebook walks through the entire process:

1.  **Setup:** Installs all necessary libraries, including `unsloth`, `transformers`, `peft`, and `trl`.
2.  **Model Loading:** Loads the Qwen3-14B model using 4-bit quantization to drastically reduce memory requirements.
3.  **Data Formatting:** The `pubmed_qa` dataset is transformed into a conversational format that the model can understand, using the Qwen3 chat template.
4.  **Training:** The model is fine-tuned using the `SFTTrainer` on the prepared dataset for 300 steps for demonstration.
5.  **Inference:** Demonstrates how to use the fine-tuned model to get direct, explanatory answers to new biomedical questions.
6.  **Saving to Hub:** Shows the correct procedure for saving the lightweight LoRA adapters to the Hugging Face Hub.

## How to Use

1.  **Open in Colab:**
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huseyincavusbi/Qwen3_BioMed_QA/blob/main/Qwen3_(14B)_PubMed_QA.ipynb)


2.  **Enable GPU:** In the Colab notebook, go to `Runtime` > `Change runtime type` and select `T4 GPU` as the hardware accelerator.

3.  **Run All Cells:** Execute the cells in the notebook from top to bottom.

4.  **Push to Your Hub (Optional):** If you wish to save your trained adapters, create a Hugging Face account, generate an access token with `write` permissions, and add it to your Colab Secrets as `HF_TOKEN`. Then, update the repository name in the final cells and run them.

## Results

After fine-tuning, the model is capable of accurately answering questions and providing context-aware explanations, as shown in the inference section of the notebook. The entire process, from setup to a trained model, can be completed in approximately 2 hours on a single T4 GPU.
