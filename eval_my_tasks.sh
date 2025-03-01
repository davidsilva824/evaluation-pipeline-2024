#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)

python -m lm_eval --model hf-mlm \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks blimp_irregular_past_participle_verbs,blimp_irregular_plural_subject_verb_agreement_1,blimp_irregular_plural_subject_verb_agreement_2,blimp_wh_island,blimp_adjunct_island,blimp_complex_NP_island,blimp_sentential_subject_island,blimp_regular_plural_subject_verb_agreement_1,blimp_regular_plural_subject_verb_agreement_2 \
    --device cuda:1 \
    --batch_size 1 \
    --log_samples \
    --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json

# Use `--model hf-mlm` and `--model_args pretrained=$MODEL_PATH,backend="mlm"` if using a custom masked LM.
# Add `--trust_remote_code` if you need to load custom config/model files.
