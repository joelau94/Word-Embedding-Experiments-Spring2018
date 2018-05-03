# Deep SRL (v180503)

3 types of GEMB model: context, char, and mix(context+char)

### Entry points:
+ Train SRL model: ```scripts/run_train.sh```
+ Train GEMB: ```scripts/run_train_gemb.sh <gemb-type>```, where gemb-type = {'context', 'char', 'mix'}.
+ Prediction: ```scripts/run_gemb_predict_conll2012.sh <gemb-type>```, where gemb-type = {'context', 'char', 'mix'}.