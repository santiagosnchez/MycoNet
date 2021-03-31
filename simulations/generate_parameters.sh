#!/bin/bash

#                  multiply fraction of spp for samples       print layout 
parallel --keep 'i=$( echo "{2}*{3}" | bc | sed "s/\..*//"); echo {1}_{2}_${i}_{4}_{5}' \
::: 500 1000 ::: 10000 100000 1000000 ::: 0.1 0.01 ::: 0.05 0.1 ::: 0.001 0.01
#   sequence       number of           fraction of     between spp     within spp
#   length         sequences           species         divergence      variation
