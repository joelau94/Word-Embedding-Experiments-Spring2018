GPU=$1

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib
export CUDA_VISIBLE_DEVICES=$GPU

CONFIG="config/srl_small_config.json"
MODEL="conll2012_model"
TRAIN_PATH="data/srl/conll2012.train.txt"
DEV_PATH="data/srl/conll2012.devel.txt"
GOLD_PATH="data/srl/conll2012.devel.props.gold.txt"

THEANO_FLAGS="mode=FAST_RUN,device=gpu$GPU,floatX=float32" python python/train.py \
   --config=$CONFIG \
   --model=$MODEL \
   --train=$TRAIN_PATH \
   --dev=$DEV_PATH \
   --gold=$GOLD_PATH
