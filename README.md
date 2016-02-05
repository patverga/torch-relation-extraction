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
Your relation data should be 4 col tsv.

entity1 \t entity2 \t relation \t 1

`./bin/process/process-data.sh -i your-data -o your-data.torch -v vocab-file`

There are other flags in you can look at by doing `./bin/process/process-data.sh --help`

For arbitrary 3 column data, use the -b flag   
If you want your rows and columns to share the same vocabulary, use the -g flag   

Training Models
------------
You can run various Universal Schema models located in [src](https://github.com/patverga/torch-relation-extraction/blob/master/src/). Check out the various options in [CmdArgs.lua](https://github.com/patverga/torch-relation-extraction/blob/master/src/CmdArgs.lua)

You can train models using this [train script](https://github.com/patverga/torch-relation-extraction/blob/master/bin/train/train-model.sh). The script takes two parameters, a gpuid (-1 for cpu) and a [config file](https://github.com/patverga/torch-relation-extraction/tree/master/bin/train/configs). You can run an example base Universal Schema model and evaluate MAP with the following command. 

`./bin/train/train-model.sh 0 bin/train/configs/uschema-example`

Evaluation
---------

#### MAP
MAP will be calculated every kth iteration based on the -evaluateFrequency cmd arg. AP is calculated on a per-column basis and then averaged to get MAP. To calculate MAP for your model, you need to generate one file per test column in the same format as your test data. Unlike the training data, in the test data you need to explicitly give negative examples. Negative samples should just have a 0 in the last column of the file while positive examples have a 1.

Place all of these files in a directory, test-data-dir for example, and then run the following command:   
`./bin/process/process-test-data-dir.sh test-data-dir test-data-dir.torch vocab-file`   
Here vocab-file should be the same vocab file that you generated your training data with.

####  [TAC slot filling task](http://www.nist.gov/tac/2013/KBP/)
- This requires setting up [Relation Factory](https://github.com/beroth/relationfactory) and setting $TAC_ROOT=/path/to/relation-factory. Just follow the setup instructions on the relation factory github or run `$TH_RELEX_ROOT/setup-relationfactory.sh`.

First run :`./setup-tac-eval.sh` 

We include candidate files for years 2012, 2013, and 2014 as well as [config files](https://github.com/patverga/torch-relation-extraction/tree/master/bin/tac-evaluation/configs/2013) to evaluate each year. 

You can tune thresholds on year 2012 and evaluate on year 2013 with this command :

`./bin/tac-evaluation/tune-and-score.sh 2012 2013 trained-model vocab-file.txt gpu-id max-length-seq-to-consider output-dir`

Relation Extraction
----------
You can also use this code to score relations. Here we'll walk through the steps to train a universal schema model. 

| e1         | e2            | relation  | 1 | 
| ------------- |:-------------:| -----| --- | 
| /m/02k__v | /m/01y5zy | $ARG1 lives in the city of $ARG2 | 1 | 
| /m/09cg6 | /m/0r297 | $ARG2 is a type of $ARG1 | 1 | 
| /m/02mwx2g | /m/02lmm0_ | /biology/gene_group_membership/gene | 1 | 
| /m/0hqv6zr | /m/0hqx04q | /medicine/drug_formulation/formulation_of | 1 | 
| /m/011zd3 | /m/02jknp | /people/person/profession | 1 | 
1. First create a training set that combines KB triples that you care about as well as text relations you care about. For example generate a file like the one above called train.tsv.
2. Next, process that file : `./bin/process/process-data.sh -i train.tsv -o data/train.torch -v vocab-file`
3. Now we want to train a model. Edit the [example lstm config](bin/train/configs/lstm-example) to say `export TRAIN_FILE=train-mtx.torch` and start the model training :  `./bin/train/train-model.sh 0 bin/train/configs/lstm-example`. This will save a model to models/lstm-example/*-model every 3 epochs.
4. Now we can use this model to perform relation extraction. Generate a candidate file called candidates.tsv. The file should be tab serparated with the following form :   
entity_1 &nbsp;&nbsp;&nbsp;&nbsp; kb_relation&nbsp;&nbsp;&nbsp;&nbsp;entity_2 &nbsp;&nbsp;&nbsp;&nbsp; doc_info &nbsp;&nbsp;&nbsp;&nbsp; arg1_start_token_idx	&nbsp;&nbsp;&nbsp;&nbsp; arg1_send_token_idx &nbsp;&nbsp;&nbsp;&nbsp;	arg2_start_token_idx &nbsp;&nbsp;&nbsp;&nbsp;	arg2_end_token_idx &nbsp;&nbsp;&nbsp;&nbsp; sentence.   
A concrete example is :   
Barack Obama &nbsp;&nbsp;&nbsp;&nbsp;	per:spouse &nbsp;&nbsp;&nbsp;&nbsp;	Michelle Obama &nbsp;&nbsp;&nbsp;&nbsp;	doc_info &nbsp;&nbsp;&nbsp;&nbsp;	0	&nbsp;&nbsp;&nbsp;&nbsp; 2 &nbsp;&nbsp;&nbsp;&nbsp;	8	 &nbsp;&nbsp;&nbsp;&nbsp; 10  &nbsp;&nbsp;&nbsp;&nbsp; Barack Obama was seen yesterday with his wife Michelle Obama .   
5. Finally, we can score each relation with the following command `th src/eval/ScoreCandidateFile.lua -candidates candidates.tsv -outFile scored-candidates.tsv -vocabFile vocab-file-tokens.txt -model models/lstm-example/5-model -gpuid 0`

This will generate a scored candidate file with the same number of lines and the sentenece replaced by a score where higher is more probable.  

Barack Obama &nbsp;&nbsp;&nbsp;&nbsp;	per:spouse &nbsp;&nbsp;&nbsp;&nbsp;	Michelle Obama &nbsp;&nbsp;&nbsp;&nbsp;	doc_info &nbsp;&nbsp;&nbsp;&nbsp;	0	&nbsp;&nbsp;&nbsp;&nbsp; 2 &nbsp;&nbsp;&nbsp;&nbsp;	8	 &nbsp;&nbsp;&nbsp;&nbsp; 10  &nbsp;&nbsp;&nbsp;&nbsp; 0.94 .
