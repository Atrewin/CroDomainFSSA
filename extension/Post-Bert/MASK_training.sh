#!/usr/bin/env bash
# -----------------------------------TODO MASK Task --------------------------------------

TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=8        # Number of sequences per batch (batch size)
UPDATE_FREQ=8          # Increase the batch size 8x total bsz = 8*8
MAX_EPOCH=500           # update epochs
SAVE_INTERVAL=50
# 三个会随着domain_A --> domain_B 变化的参数
domains=book2dvd
DATA_DIR=data/domain_data/data-bin/${domains}/mask


SAVE_PATH=checkpoints/MASK_${domains}                       # 不知道为什么，无法设置 @jinhui 0315 因为做了代换为 --save-dir 而不是看到的--save_dir

CUDA_VISIBLE_DEVICES=3 fairseq-train --fp16 $DATA_DIR \
    --save-dir $SAVE_PATH \
    --save-interval $SAVE_INTERVAL \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
    --skip-invalid-size-inputs-valid-test \
    --max-epoch $MAX_EPOCH \
    --reset-optimizer --reset-dataloader --reset-meters \



