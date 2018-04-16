export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

CONFIG="config/srl_config.json"
MODEL="conll2012_model"
GEMB_MODEL="conll2012_gemb"
TRAIN_PATH="data/srl/conll2012.train.txt"
DEV_PATH="data/srl/conll2012.devel.txt"
GOLD_PATH="data/srl/conll2012.devel.props.gold.txt"

#THEANO_FLAGS="mode=FAST_RUN,device=gpu$1,floatX=float32,lib.cnmem=0.9" python python/train_gemb.py \
THEANO_FLAGS="mode=FAST_RUN,device=cpu" python python/train_gemb.py \
   --config=$CONFIG \
   --model=$MODEL \
   --gemb-model=$GEMB_MODEL \
   --train=$TRAIN_PATH \
   --dev=$DEV_PATH \
   --gold=$GOLD_PATH
