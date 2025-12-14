#!/bin/bash
cat SH_at_least_5_seqs.txt | parallel 'grep -A1 {} UNITE_public_04.02.2020.fasta ' | sed '/^--$/d'
