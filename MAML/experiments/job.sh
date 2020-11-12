#!/usr/bin/env bash

#model_domain_expand_${aux}_10_10_meta.mdl $1
#model_4zeroshot_${aux}_10_10_${1}_meta.mdl
for ((aux=0; aux <= 104 ; aux++))
do
        sbatch --job-name=${2}-${aux}  --output=out-${2}-${aux}-fewshot.txt --error=err-${2}-${aux}-fewshot.txt  transformer_bi_auxs.slurm   $1 1 $2 $aux
done