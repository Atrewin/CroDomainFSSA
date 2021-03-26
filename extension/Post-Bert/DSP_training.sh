#!/usr/bin/env bash

## -----------------------------------TODO DSP Task--------------------------------------
#
TOTAL_NUM_UPDATES=7812  # 10 epochs through IMDB for bsz 32
WARMUP_UPDATES=469      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
HEAD_NAME=dsp_head     # Custom name for the classification head.
NUM_CLASSES=2           # Number of classes for the classification task.
MAX_SENTENCES=8         # Batch size.
UPDATE_FREQ=8          # Increase the batch size 8x total bsz = 8*8
MAX_EPOCH=25            # update epochs

# 三个会随着domain_A --> domain_B 变化的参数
ROBERTA_PATH=/home/cike/project/fairseq/extension/RoBERT/pre-train/checkpoints/MASK_book2kitchen/checkpoint_last.pt # 取到MASK训练后的checkpoint
DATA_DIR=data/domain_data/data-bin/book2kitchen/DSP                   # 二进制数据所在路径，注意上下文路径好像人家的有input0
SAVE_PATH=checkpoints/DSP_book2kitchen                        # 不知道为什么，无法设置 @jinhui 0315 实际上，它会做一次变换应该是 --save-dir

CUDA_VISIBLE_DEVICES=4 fairseq-train $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --save-dir $SAVE_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch $MAX_EPOCH \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test