#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module purge   # clear any inherited modules
module load python-data
module load pytorch/1.3.0

export PYTHONPATH="${PYTHONPATH}:/scratch/project_2000582/XDL-Adaptation/"
export GLUE_DIR=/scratch/project_2000582/transformers/glue_data
export TASK_NAME=xnli
export MODEL_DIR=/scratch/project_2000582/XDL-Adaptation/data/BERT/meta_learning/mnli_bert_multi_cased
srun python maml_bert.py \
    --model_type bert \
    --cache_dir $MODEL_DIR\
    --model_name_or_path bert-multi-cased \
    --task_name $TASK_NAME \
    --do_maml \
    --do_fine_tune_task \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 1e-5 \
    --num_train_epochs 3\
    --meta_learn_iter=$1 \
    --inner_train_steps=$2\
    --seed=$3\
    --percent=100 \
    --auxiliary=$4 \
    --bi_auxs\