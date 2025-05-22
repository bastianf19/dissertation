# from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


from helper import clean_text, read_data, write_data



def main():
    # params
    input_filepath = '../dataset/translated/opus-mt/babylm_data_id/babylm_10M/cbt.train'
    output_filepath = '../dataset/clean_data/babylm_10M/cbt.train'
    
    # load data, cleaning
    texts = read_data(input_file=input_filepath)
    print(f'sampled texts: {texts[:2]}')
    
    cleaned_text = []
    for t in texts:
        cleaned_text.append(clean_text(text=t) + '\n')
        
    print(f'sampled cleaned texts: {cleaned_text[:2]}')
    
    write_data(input_text=cleaned_text, output_file=output_filepath)

if __name__ == '__main__':
    main()
    