
for ((i=1;i<=4;i++));
do

      ## -----------------------------------TODO DSP Task--------------------------------------
  #
  TOTAL_NUM_UPDATES=7812  # 10 epochs through IMDB for bsz 32
  WARMUP_UPDATES=469      # 6 percent of the number of updates
  LR=1e-05                # Peak LR for polynomial LR scheduler.
  HEAD_NAME=dsp_head     # Custom name for the classification head.
  NUM_CLASSES=2           # Number of classes for the classification task.
  MAX_SENTENCES=8         # Batch size.
  UPDATE_FREQ=8          # Increase the batch size 8x total bsz = 8*8
  MAX_EPOCH=10           # update epochs $(($i*5))
  SAVE_INTERVAL=9
  # 三个会随着domain_A --> domain_B 变化的参数
  ROBERTA_PATH=checkpoints/MASK_${domains}/checkpoint_last.pt # 取到MASK训练后的checkpoint
  DATA_DIR=data/domain_data/data-bin/book2kitchen/DSP                   # 二进制数据所在路径，注意上下文路径好像人家的有input0
  SAVE_PATH=checkpoints/DSP_${domains}                      # 不知道为什么，无法设置 @jinhui 0315 实际上，它会做一次变换应该是 --save-dir

  CUDA_VISIBLE_DEVICES=4 fairseq-train $DATA_DIR \
      --restore-file $ROBERTA_PATH \
      --save-dir $SAVE_PATH \
      --save-interval $SAVE_INTERVAL \
      --max-positions 512 \
      --batch-size $MAX_SENTENCES \
      --max-tokens 4400 \
      --task sentence_prediction \
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
      --skip-invalid-size-inputs-valid-test \
      --reset-optimizer --reset-dataloader --reset-meters

  # -----------------------------------TODO MASK Task --------------------------------------
  TOTAL_UPDATES=125000    # Total number of training steps
  WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
  PEAK_LR=0.0005          # Peak learning rate, adjust as needed
  TOKENS_PER_SAMPLE=512   # Max sequence length
  MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
  MAX_SENTENCES=8        # Number of sequences per batch (batch size)
  UPDATE_FREQ=8          # Increase the batch size 8x total bsz = 8*8
  MAX_EPOCH=70         # update epochs$(($i*20))
  SAVE_INTERVAL=$(($i+50))
  # 三个会随着domain_A --> domain_B 变化的参数
  domains=book2dvd

  DATA_DIR=data/domain_data/data-bin/${domains}/mask
  ROBERTA_PATH=checkpoints/DSP_${domains}/checkpoint_last.pt #取到上一个训练的模式
  SAVE_PATH=checkpoints/MASK_${domains}                       # 不知道为什么，无法设置 @jinhui 0315 因为做了代换为 --save-dir 而不是看到的--save_dir

  CUDA_VISIBLE_DEVICES=4 fairseq-train --fp16 $DATA_DIR \
      --restore-file $ROBERTA_PATH \
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
      --reset-optimizer --reset-dataloader --reset-meters



done


# -----------------------------------TODO MASK Task --------------------------------------
  TOTAL_UPDATES=125000    # Total number of training steps
  WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
  PEAK_LR=0.0005          # Peak learning rate, adjust as needed
  TOKENS_PER_SAMPLE=512   # Max sequence length
  MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
  MAX_SENTENCES=8        # Number of sequences per batch (batch size)
  UPDATE_FREQ=8          # Increase the batch size 8x total bsz = 8*8
  MAX_EPOCH=100         # update epochs$(($i*20))
  SAVE_INTERVAL=60
  # 三个会随着domain_A --> domain_B 变化的参数
  domains=book2dvd

  DATA_DIR=data/domain_data/data-bin/${domains}/mask
  ROBERTA_PATH=checkpoints/DSP_${domains}/checkpoint_last.pt #取到上一个训练的模式
  SAVE_PATH=checkpoints/MASK_${domains}                       # 不知道为什么，无法设置 @jinhui 0315 因为做了代换为 --save-dir 而不是看到的--save_dir

  CUDA_VISIBLE_DEVICES=4 fairseq-train --fp16 $DATA_DIR \
      --restore-file $ROBERTA_PATH \
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
      --reset-optimizer --reset-dataloader --reset-meters