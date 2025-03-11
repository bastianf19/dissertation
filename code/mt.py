# Load model directly
#%%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#%%
tokenizer = AutoTokenizer.from_pretrained("../models/opus-mt-en-id")
model = AutoModelForSeq2SeqLM.from_pretrained("../models/opus-mt-en-id")
#%%
with open('../dataset/babylm_data/babylm_10M/aochildes.train', 'rb') as f:
    aochildes = f.read()
aochildes[:2]
# %%
