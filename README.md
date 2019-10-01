# Identification of Verbal Multiword Expressions Using Deep Learning Architectures and Representation Learning Methods

- This repository contains the source code of my masters research. The thesis report is available the following link. The first three parts of my thesis are developed under the Deep-BGt research project and the details are available in the following repository. This repo contains the details of the representation learning methods for VMWE identification.  
- I used the fastText word embeddings and the bigappy-unicrossy tagging scheme.
-  Bulgarian (BG), German (DE), Greek (EL), English (EN), Spanish (ES), Basque (EU), Farsi (FA), French (FR),
          Hebrew (HE), Hindu (HI), Crotian (HR), Hungarian (HU), Italian (IT), Lithuanian (LT),
           Polish (PL), Portuguese (PT), Romanian (RO), Slovenian (SL), and Turkish (TR).

## Requirements

Setup with virtual environment (Python 3):
-  python3 -m venv vmwe_rl_venv
   source vmwe_rl_venv/bin/activate
- Install the requirements:
   vmwe_rl_venv/bin/pip3 install -r requirements.txt

## Example Usage:
- The following guide show an example usage of the model for English with bigappy-unicrossy tagging scheme.
- Running all experiments for 19 languages with the three different tagging schemes -IOB, gappy-1-level, and bigappy-unicrossy- takes at least one week.
- Since English is one the smallest corpus, we choose this example to show our new tagging scheme.
- Running this experiment will take approximately a few hours.
- Instructions
      1. Download word embeddings: " wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz "
         Locate it into CICLing_42/input/embeddings
      2. Download the PARSEME Corpora: " wget https://gitlab.com/parseme/sharedtask-data/-/archive/master/sharedtask-data-master.zip "
         Unzip the downloaded file
         Locate it into CICLing_42/input/corpora
      3. Change directory to the location of the source code which is CICLing_42
      4. Run the instructions in "Setup with virtual environment (Python 3)"
      5. Run the command to train the model: python3 Runner.py -l EN -t gappy-crossy
         If you want to try the model with another configuration, change language code after -l, and tag after -t
         Languages: BG, DE, EL, EN, ES, EU, FA, FR, HE, HI HR, HU, LT, IT, PL, PT, RO, SL, TR
         Tags: IOB, gappy-1, gappy-crossy
      6. Open the file in CICLing_42/eval.cmd, copy and run the command in this file to evaluate the accuracy
      7. The results will be in CICLing_42/output/EN/eval.txt
