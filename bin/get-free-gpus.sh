#!/usr/bin/env bash

if ! type "nvidia-smi" &> /dev/null; then
  echo "-1"
else
  nvidia-smi -q -d UTILIZATION | grep Gpu | awk BEGIN{'print "-1"; i=0;'}{'if($3 == 0) print i; i++;'}
fi
