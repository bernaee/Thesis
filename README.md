# Identification of Verbal Multiword Expressions Using Deep Learning Architectures and Representation Learning Methods

- This repository contains the source code of my master research. The thesis report is available at the following link: https://drive.google.com/file/d/15_Vi6RJ0Zy_m-Yln9d4LnBISkHVfXWW5/view . 
- The details of the first three parts of the thesis are available in the following repository: https://github.com/deep-bgt/Deep-BGT. 
- This repo contains the details of the representation learning methods for VMWE identification. I compare character-level CNNs and character-level BiLSTM networks. Also, I analyze two different input schemes to represent morphological information using BiLSTM networks.
- I use the PARSEME VMWE Corpora Edition 1.1, the fastText word embeddings and the bigappy-unicrossy tagging scheme.
- The corpora covers 19 languages as follows:
Bulgarian (BG), German (DE), Greek (EL), English (EN), Spanish (ES), Basque (EU), Farsi (FA), French (FR),
          Hebrew (HE), Hindu (HI), Crotian (HR), Hungarian (HU), Italian (IT), Lithuanian (LT),
           Polish (PL), Portuguese (PT), Romanian (RO), Slovenian (SL), and Turkish (TR).

## Requirements

Setup with virtual environment (Python 3):
-  python3 -m venv vmwe_rl_venv
   source vmwe_rl_venv/bin/activate
- Install the requirements:
   vmwe_rl_venv/bin/pip3 install -r requirements.txt

## Usage:
- 
