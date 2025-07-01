from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFD, StripAccents, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import glob
from transformers import PreTrainedTokenizerFast, AutoTokenizer

def train_tokenizer():
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), StripAccents()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=50257,
        min_frequency=2,
        show_progress=True,
        special_tokens=[
            "<|endoftext|>",
            "[PAD]",
        ],
        initial_alphabet=ByteLevel.alphabet()
    )
    files = glob.glob("/home/s2587034/dissertation/dataset/clean_data/nllb/babylm_data_ind/babylm_100M/*.train")
    tokenizer.train(files, trainer)
    tokenizer.decoder = ByteLevelDecoder()

    # Wrap & save in HF format
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|endoftext|>",
        pad_token="[PAD]",
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>"
    )
    # save tokenizer
    save_dir = "/home/s2587034/dissertation/models/indo_tokenizer"
    hf_tok.save_pretrained(save_dir)

def test_tokenizer():
    try:
        # tokenizer = AutoTokenizer.from_pretrained("/home/s2587034/dissertation/models/indo_tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        test_sentences = [
                "Kau ingin ini kembali ke tanganmu.",
                "Kau ingin itu di tanganku mungkin.",
                "Mari kita buka jari-jarimu sedikit.",
                "Dapatkah aku membantumu dengan membuka jari-jarimu?"
            ]

        print("\n===== Tokenizer Test Output =====")
        for sent in test_sentences:
            # Get the encoded output
            encoded = tokenizer(sent, return_tensors=None)
            tokens = tokenizer.tokenize(sent)
            
            print("\nOriginal:", sent)
            print("Tokens:", tokens)
            print("Token IDs:", encoded['input_ids'])
            print("Attention Mask:", encoded.get('attention_mask', 'N/A'))
            print("Decoded:", tokenizer.decode(encoded['input_ids']))

        print("\n===== All tests completed successfully =====")
        
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise e


if __name__ == "__main__":
    # train_tokenizer()
    test_tokenizer()