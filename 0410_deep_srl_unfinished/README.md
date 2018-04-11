# Deep SRL (v180410)

### Major changes compared to original code:
+ ```python/train_gemb.py```: train GEMB params
+ ```python/predict_gemb.py```: predict with GEMB replacing OOV embeddings
+ ```python/neural_srl/theano/gemb.py```: define computation graphs for GEMB
+ ```python/neural_srl/theano/tagger.py```: method ```add_gemb``` attach new attributes to tagger model
+ ```python/neural_srl/shared/tagger_data.py```: added several methods: formatting data for GEMB training and prediction
+ ```python/neural_srl/shared/reader.py```: added several methods: loading data for GEMB training and prediction

### Entry points:
+ Train SRL model: ```scripts/run_train.sh```
+ Train GEMB: ```scripts/run_train_gemb.sh```
+ Prediction: ```scripts/run_gemb_predict_conll2012.sh```