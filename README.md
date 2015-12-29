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
If you specify a test file (or comma seperated list of files), MAP will be calculated every kth iteration where k is a cmd arg set as -evaluateFrequency k. Each should be its own relation and will have its average precision calculated. MAP will be calculated as the average over all the files.

There are also scripts to evaluate using the [TAC slot filling task](http://www.nist.gov/tac/2013/KBP/). This requires setting up [Relation Factory](https://github.com/beroth/relationfactory) and setting TAC_ROOT. If you follow the instructions, the candidate file will be located at /your/rundir/candidates. You can then create a config file similar to [this one](https://github.com/patverga/torch-relation-extraction/tree/master/bin/tac-evaluation/configs/2013) that points to your run directory, candidate file, etc. You can then evaluate using [this script](https://github.com/patverga/torch-relation-extraction/blob/master/bin/tac-evaluation/tac-eval-score-tuned.sh). You should tune the thresholds using [this script](https://github.com/patverga/torch-relation-extraction/blob/master/bin/tac-evaluation/tac-eval-tune-thresh.sh) first using dev data. If you dont want to, you can use [this no-threshold param file](https://github.com/patverga/torch-relation-extraction/blob/master/bin/tac-evaluation/no-thresh-params). 

`./bin/tac-evaluation/tac-eval-score-tuned.sh your-config trained-model vocab-file.txt gpu-id max-length-seq-to-consider "" bin/tac-evaluation/no-thresh-params output-dir`
