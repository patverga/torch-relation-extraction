#!/usr/bin/env bash

# candidate files are too big for git, untar them
cd ${TH_RELEX_ROOT}/data
tar -xvzf candidates.tar.gz

# make the various scoring scripts
for y in 2 3 4; do
    cd ${TH_RELEX_ROOT}/bin/tac-evaluation/eval-scripts/SFScore-201${y}
    javac  SFScore.java
done

cd ${TH_RELEX_ROOT}
