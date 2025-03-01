<h1 align='center'> Recurrent Neural Network Acoustic Models for Speech Recognition 🗣️ </h1>

This project explores the use of Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks, for speech recognition tasks. The project is inspired by a Google [paper](paper.pdf) on speech recognition.    
The main goal of the Notebook is to serve as an educational resource for learning how speech-to-text systems work, from the features extraction (with the use of Mel-frequency cepstral coefficients (MFCCs) and transcription encoding), the construction of the model, the use of the Connectionist Temporal Classification (CTC) loss, the Word Error Rate (WER)...   
Watch the project [poster](poster.pdf)!   
Finally, an ASR application is in progress... 🚧

<br>

## 🌟 Features

In this notebook, we implement a speech recognition system using LSTM-based RNNs with CTC loss. The notebook is designed to be educational and will walk through several important concepts in speech processing and deep learning, such as:

- **Mel-Frequency Cepstral Coefficients (MFCCs):** Used as a feature representation for speech signals.
- **LSTM-based RNNs:** Using Long Short-Term Memory networks for sequence modeling.
- **CTC Loss (Connectionist Temporal Classification):** A loss function that enables speech-to-text models to align sequences of different lengths (speech and text).

The project is structured to help understanding the theory behind these concepts, as well as how they are implemented in practice for speech recognition tasks.

<br>

An ASR application as been added based on the [open-ai whisper](https://openai.com/index/whisper/) model. 

<br>

## 🗂️ Project Structure

Here's the current structure of the project:

```
Recurrent-Neural-Network-Acoustic-Models-for-Speech-Recognition/
├── app/                                # Raw and processed audio data 
│   ├── devices.py                      # Check available microphones 
│   ├── load.py                         # Whisper model loading 
│   ├── main.py                         # Main ASR Application 
│   └── microphone_test.py              # Test microphone 
│   
├── data/                               # Raw and processed audio data 
│   ├── audio_sample                    # Raw audio used in the Notebook 
│   │   ├── ...
│   │   └── ...
│   │ 
│   ├── test                            # Numpy array processed test audio files
│   │   ├── X_test_augmented.npy
│   │   └── X_test_augmented.npy
│   │ 
│   ├── train                           # Numpy array processed train audio files
│   │   ├── X_train_augmented.npy
│   │   └── X_train_augmented.npy
│   │ 
│   ├── test_validated.json             # Test file contenaing data path and transcript
│   └── train_validated.json            # Train file contenaing data path and transcript
│
├── img/                                # Notebook images
│   ├── ...
│   └── ...
│
├── model/                              # Trained models 
│   ├── model_weights_augmented.pth     # 'Final' RNN LSTM model
│   └── single.pth                      # Model trained with one audio file
│
├── .gitignore                          # Git ignore file
├── environment.yaml                    # Yaml file to create the environment
├── LICENSE                             # License file
├── notebook.ipynb                      # Educative notebook
├── paper.pdf                           # Paper that inspired the repo
├── poster.pdf                          # Project poster
├── README.md                           # Project documentation (this file)
└── requirements.txt                    # Python dependencies
```


<br>

## 💾 Dataset

The dataset used for this project is the [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) dataset. It is not in the [data](data) folder, as it is too heavy!

<br>

## ⚡ Quick Start

Before you can simulate the attack, you need to set up the project and configure the environment variables.


### 1. Clone the Repository 📥

```bash
git clone https://github.com/Ad882/Recurrent-Neural-Network-Acoustic-Models-for-Speech-Recognition.git
cd Recurrent-Neural-Network-Acoustic-Models-for-Speech-Recognition
```

--- 
### 2. Install Dependencies 🧑‍💻

Make sure you have Python 3.7+ installed. Then, install the necessary dependencies with:

```bash
pip install -r requirements.txt
```

It is possible to create a new environment with all the dependancies:

```bash
conda env create --name asr --file=environment.yaml
```
  
--- 
### 4. Run the Notebook 🚀

Then just run the notebook and enjoy!


<br>


## 🎙️ Application

An ASR application is being implemented...
