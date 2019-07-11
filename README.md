Identification of Verbal Multiword Expressions Using Deep Learning Architectures and Representation Learning Methods

Requirements
- Python 3.6
- Keras 2.2.4 with Tensorflow 1.12.0, and keras-contrib==2.0.8
- We cannot guarantee that the code works with different versions for Keras / Tensorflow.
- We cannot provide the data used in the experiments in this code repository, because we have no right to distribute the corpora provided by PARSEME Shared Task Edition 1.1 .
       1. Please download corpora by command " wget https://gitlab.com/parseme/sharedtask-data/-/archive/master/sharedtask-data-master.zip "
          Unzip the downloaded file
          Locate it into CICLing_42/input/corpora
       2. All word embeddings are available in the following links:
            BG: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bg.300.bin.gz
            DE: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz
            EL: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.el.300.vec.gz
            EN: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
            ES: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz
            EU: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eu.300.vec.gz
            FA: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.vec.gz
            FR: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz
            HE: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.he.300.vec.gz
            HI: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.vec.gz
            HR: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hr.300.vec.gz
            HU: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hu.300.vec.gz
            IT: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.vec.gz
            LT: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.lt.300.vec.gz
            PL: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.vec.gz
            PT: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz
            RO: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ro.300.vec.gz
            SL: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sl.300.vec.gz
            TR: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.bin.gz
          Download word embeddings by command " wget language_link "
          Locate them into CICLing_42/input/embeddings
       3. Language codes are Bulgarian (BG), German (DE), Greek (EL), English (EN), Spanish (ES), Basque (EU), Farsi (FA), French (FR),
          Hebrew (HE), Hindu (HI), Crotian (HR), Hungarian (HU), Italian (IT), Lithuanian (LT),
           Polish (PL), Portuguese (PT), Romanian (RO), Slovenian (SL), and Turkish (TR).

