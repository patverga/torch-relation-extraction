#!/bin/bash

IN=$1

paste <(cut -d'	' -f 1-2 $IN) <(cut -d'	' -f 3 $IN| awk '{ printf $1" "; for(i = 2; i <= (NF-1); i++) { printf $i"_es "; } ; print $NF}') <(cut -d'	' -f 4 $IN) > ${IN}.es_appended
