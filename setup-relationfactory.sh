#!/usr/bin/env bash

# clone relation factory repo
git clone git@github.com:beroth/relationfactory.git
cd relationfactory

# download and unpack models
wget https://www.lsv.uni-saarland.de/fileadmin/data/relationfactory_models.tar.gz
tar xzf relationfactory_models.tar.gz

# set up environment variables
echo "export TAC_ROOT=`pwd`" >> $HOME/.bash_profile
echo "export TAC_MODELS=`pwd`/relationfactory_models" >> $HOME/.bash_profile
source $HOME/.bash_profile

# compile the system
${TAC_ROOT}/bin/generate_system.sh