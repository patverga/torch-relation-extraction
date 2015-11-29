#!/bin/bash

IN=$1
TRANSLATE="/home/pat/canvas/universal-schema/univSchema/torch/data/translate-dict-en_es"

paste <(cut -d'	' -f 1-2 $IN) <(cut -d'	' -f 3 $IN| 
awk -v "T=$TRANSLATE" 'BEGIN{
      FS=OFS=" "
   
   while ( (getline line < T) > 0 ) {
      split(line,f)
      map[f[2]] = f[1]
   } 
}
{printf $1" "; for(i = 2; i <= (NF-1); i++) {
t=map[$i];
if (t~/#/ || length(t) > 0){printf t" "} else {printf $i"_es "} ; 
} ; print $NF"\t1"}') > ${IN}.es_appended-and-translated
