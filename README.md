# Prerequisites
python 3.7.12  
pytorch 1.8.1  
cuda 10.2  
transformers 4.12.5  
json 2.0.9  
numpy 1.21.2  
pandas 1.3.4  
dgl-cu102 0.6.1  
nltk 3.6.5  
# Descriptions
**data** - contains dataset.  
* ```bert-base-uncased```: put the download Pytorch bert model here (config.json, pytorch_model.bin, vocab.txt) (https://huggingface.co/bert-base-uncased/tree/main). 
* ```rr-submission```: contains RR dataset, download from (https://github.com/LiyingCheng95/ArgumentPairExtraction/tree/master/data/rr-submission) 

**saved_models** - contains saved models, training logs and results.  

**utils** - utils code.  
* ```config.py```: parameter setting. 
* ```fun.py```: functions.
* ```metrics.py```: evaluation metrics.
* ```models.py```: proposed model.

```prepare_data.py``` - prepare the input data.  
```run.py``` - train and evaluate the proposed transition-based model.  
```to_bioes.py``` - change to bioes label.  

# Usage
python prepare_data.py  
python to_bioes.py  
python run.py  
