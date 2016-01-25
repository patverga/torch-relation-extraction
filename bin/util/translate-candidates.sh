#!/bin/bash

IN=$1
OUT=$2
TRANSLATE="/home/pat/canvas/universal-schema/univSchema/torch/data/translate-dict-en_es"

paste <(cut -d'	' -f 1-8 $IN) \
<(cut -d'	' -f 9 $IN | sed 's/[0-9]/#/g' |\
awk -v "T=$TRANSLATE" 'BEGIN{
      FS=OFS=" ";   
   while ( (getline line < T) > 0 ) {
      split(line,f)
      map[f[2]] = f[1]
   } 
}
{
  printf $1" ";
 for(i = 2; i <= (NF-1); i++) {
   t=map[$i];
   if (t~/#/ || length(t) > 0){printf t" "} else {printf $i"_es "} ; 
   };
  print $NF
}') > $OUT
