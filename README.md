# torch-relation-extraction
Universal Schema based relation extraction implemented in Torch.

export TH_RELEX_ROOT=/path/to/this/proj

Data Processing
--------------
Your data should be 4 col tsv.

entity1 \t entity2 \t relation \t 1

`./bin/process/process-data.sh -i your-data -o your-data.torch`

There are other flags in you can look at by doing `./bin/process/process-data.sh --help`


Running Models
------------
You can run various Universal Schema models located in [src](https://github.com/patverga/torch-relation-extraction/blob/master/src/). Check out the various options in [CmdArgs.lua](https://github.com/patverga/torch-relation-extraction/blob/master/src/CmdArgs.lua)

`th src/UniversalSchemaLSTM.lua -gpuid 0` will train an LSTM Universal Schema model using some sample train data and do map evalution on sample test data.
