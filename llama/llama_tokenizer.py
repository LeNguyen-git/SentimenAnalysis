import llama_preprocessing

import re
import json
from collections import Counter

with open('../data/UIT-VSFC/merge_data/all_text.txt', 'r', encoding='utf-8') as f:
    raw_data = f.read().splitlines()


raw_data = [llama_preprocessing.preprocess_text(text) for text in raw_data]


class LLaMaTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)
        self.special_tokens = ['<pad>', '<unk>', '<s>', '</s>']

    def tokenizer(self,text):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [item.strip() for item in tokens if item.strip() != '']
        return tokens
    
    def build_vocab(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        counter = Counter()
        for text in texts:
            counter.update(self.tokenizer(text))

        token_set = sorted(set(counter.keys()))
        all_tokens = self.special_tokens + token_set

        # str_to_int: token (str) -> idx (int)
        self.str_to_int = {token: idx for idx, token in enumerate(all_tokens)}
        # int_to_str: idx (int) -> token (str)
        self.int_to_str = {idx: token for token, idx in self.str_to_int.items()}
        self.vocab_size = len(self.str_to_int)
        return self.str_to_int
    
    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenizer(text)
        if add_special_tokens:
            tokens = ['<s>'] + tokens + ['</s>']

        unk_id = self.str_to_int.get('<unk>')
        if unk_id is None:
            raise ValueError("Bạn chưa thêm token <unk> vào vocab. Vui lòng build_vocab đúng cách.")

        return [self.str_to_int.get(token, self.str_to_int['<unk>']) for token in tokens]
    
    def decode(self, indices, skip_pad=True):
        tokens = [self.int_to_str.get(idx, '<unk>') for idx in indices]
        if skip_pad:
            tokens = [tok for tok in tokens if tok != '<pad>']
        return ' '.join(tokens)
    
    def save_vocab(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.str_to_int, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.str_to_int = vocab
        self.int_to_str = {int(idx) : token for token, idx in vocab.items()}
        self.vocab_size = len(vocab)
    
tokenizer = LLaMaTokenizer({})
tokenizer.build_vocab(raw_data)
# tokenizer.save_vocab('../data/UIT-VSFC/llama_vocab.json')

# print("Build vocab thành công")
# print("Số lượng từ trong vocab:", tokenizer.vocab_size)
# print("Vocab:", tokenizer.str_to_int)


# print("------------------------------------------------------------------------------------------------------------")

# encode = tokenizer.encode("giáo viên hôm nay không ra thêm bài tập về nhà, thiệt luôn á hả", add_special_tokens=True)
# print("Encoded:", encode)

# decode = tokenizer.decode(encode, skip_pad=True)
# print("Decoded:", decode)

# print("------------------------------------------------------------------------------------------------------------")


