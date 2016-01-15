# Universal Schema based relation extraction implemented in Torch.

Paper
------------
This code was used for the paper [Multilingual Relation Extraction using Compositional Universal Schema](http://arxiv.org/abs/1511.06396) by Patrick Verga, David Belanger, Emma Strubell, Benjamin Roth, Andrew McCallum.

If you use this code, please cite us.


Dependencies
-----------
- [torch](https://github.com/torch/torch7)
- [nn](https://github.com/torch/nn)
- [rnn](https://github.com/Element-Research/rnn)
- [optim](https://github.com/torch/optim)
- set this environment variable : TH_RELEX_ROOT=/path/to/this/proj


Data Processing
--------------
Your data should be 4 col tsv.

entity1 \t entity2 \t relation \t 1

`./bin/process/process-data.sh -i your-data -o your-data.torch -v vocab-file`

There are other flags in you can look at by doing `./bin/process/process-data.sh --help`


Training Models
------------
You can run various Universal Schema models located in [src](https://github.com/patverga/torch-relation-extraction/blob/master/src/). Check out the various options in [CmdArgs.lua](https://github.com/patverga/torch-relation-extraction/blob/master/src/CmdArgs.lua)

You can train models using this [train script](https://github.com/patverga/torch-relation-extraction/blob/master/bin/train/train-model.sh). The script takes two parameters, a gpuid (-1 for cpu) and a [config file](https://github.com/patverga/torch-relation-extraction/tree/master/bin/train/configs). You can run an example base Universal Schema model and evaluate MAP with the following command. 

`./bin/train/train-model.sh 0 bin/train/configs/uschema-example`

Evaluation
---------

#### MAP
If you specify a test file (or comma seperated list of files), MAP will be calculated every kth iteration where k is a cmd arg set as -evaluateFrequency k. Each should be its own relation and will have its average precision calculated. MAP will be calculated as the average over all the files.

####  [TAC slot filling task](http://www.nist.gov/tac/2013/KBP/)
- This requires setting up [Relation Factory](https://github.com/beroth/relationfactory) and setting $TAC_ROOT=/path/to/relation-factory. Just follow the setup instructions on the relation factory github, its easy.

First run :`./setup-tac-eval.sh` 

We include candidate files for years 2012, 2013, and 2014 as well as [config files](https://github.com/patverga/torch-relation-extraction/tree/master/bin/tac-evaluation/configs/2013) to evaluate each year. 

You can tune thresholds on year 2012 and evaluate on year 2013 with this command :

`./bin/tac-evaluation/tune-and-score.sh 2012 2013 trained-model vocab-file.txt gpu-id max-length-seq-to-consider output-dir`
