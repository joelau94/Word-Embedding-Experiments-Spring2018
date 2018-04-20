#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

MODEL_PATH="./conll2012_model"
GEMB_MODEL_PATH="./conll2012_model"
INPUT_PATH="data/srl/conll2012.test.txt"
GOLD_PATH="data/srl/conll2012.test.props.gold.txt"
OUTPUT_PATH="temp/conll2012.test.out"

if [ "$#" -gt 0 ]
then
THEANO_FLAGS="optimizer=fast_compile,floatX=float32,device=gpu$1,lib.cnmem=0.8" python python/predict_gemb.py \
  --model="$MODEL_PATH" \
  --gemb-model="$GEMB_MODEL_PATH" \
  --input="$INPUT_PATH" \
  --output="$OUTPUT_PATH" \
  --gold="$GOLD_PATH"
else
THEANO_FLAGS="optimizer=fast_compile,floatX=float32" python python/predict_gemb.py \
  --model="$MODEL_PATH" \
  --gemb-model="$GEMB_MODEL_PATH" \
  --input="$INPUT_PATH" \
  --output="$OUTPUT_PATH" \
  --gold="$GOLD_PATH"
fi

