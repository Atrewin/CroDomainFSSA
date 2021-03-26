#!/usr/bin/env bash



# -----------------------------------TODO cross domain--------------------------------------




### TODO DSP MASK
#
for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs ../domain_data/processed_data/book2kitchen/MASK/${SPLIT}.raw \
        --outputs ../domain_data/processed_data/book2kitchen/MASK/${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done

### Finally preprocess/binarize the data using the GPT-2 fairseq dictionary:

fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref ../domain_data/processed_data/book2kitchen/MASK/train.bpe \
    --validpref ../domain_data/processed_data/book2kitchen/MASK/valid.bpe \
    --testpref ../domain_data/processed_data/book2kitchen/MASK/test.bpe \
    --destdir ../domain_data/data-bin/book2kitchen/mask \
    --workers 60


#### BPE encode
##
## TODO DSP Task
#for SPLIT in train dev; do
#    python -m examples.roberta.multiprocessing_bpe_encoder \
#        --encoder-json gpt2_bpe/encoder.json \
#        --vocab-bpe gpt2_bpe/vocab.bpe \
#        --inputs "../domain_data/processed_data/book2kitchen/DSP/$SPLIT.input0" \
#        --outputs "../domain_data/processed_data/book2kitchen/DSP/$SPLIT.input0.bpe" \
#        --workers 60 \
#        --keep-empty
#done
#
### Preprocess data
## doc
#fairseq-preprocess \
#    --only-source \
#    --trainpref "../domain_data/processed_data/book2kitchen/DSP/train.input0.bpe" \
#    --validpref "../domain_data/processed_data/book2kitchen/DSP/dev.input0.bpe" \
#    --destdir "../domain_data/data-bin/book2kitchen/DSP/input0" \
#    --workers 60 \
#    --srcdict gpt2_bpe/dict.txt
#
## label
#fairseq-preprocess \
#    --only-source \
#    --trainpref "../domain_data/processed_data/book2kitchen/DSP/train.label" \
#    --validpref "../domain_data/processed_data/book2kitchen/DSP/dev.label" \
#    --destdir "../domain_data/data-bin/book2kitchen/DSP/label" \
#    --workers 60