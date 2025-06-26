import os
import json
import re
from collections import defaultdict
from preprocessing import TextPreprocessor

class BertTokenizer:
    def __init__(self, vocab_size=None,  min_frequency=None):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocab = {}

        self.special_tokens = {
                    '[PAD]': 0,
                    '[UNK]': 1,
                    '[CLS]': 2,
                    '[SEP]': 3,
                    '[MASK]': 4,
                }
        
        self.subwords = set()
        self.merges = []

    def get_word_frequency(self, corpus):
        word_freq = defaultdict(int)

        for words, freq in corpus.items():
            # symbols = re.findall(r'\w+', words)
            symbols = words.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                word_freq[pair] += freq
        return word_freq

    def merge_vocab(self, word_freq, corpus):
        bigram = ' '.join(word_freq)
        replacement = ''.join(word_freq)

        new_vocab = defaultdict(int)

        for words in corpus:
            new_words = words.replace(bigram, replacement)
            new_vocab[new_words] += corpus[words]

        return new_vocab
    
    def build_vocab(self, texts):
        corpus = defaultdict(int)


        for text in texts:
            if not text.strip():
                continue
            for word in text.strip().split():
                chars = ' '.join(list(word)) + ' </w>'
                corpus[chars] += 1

        vocab = dict(self.special_tokens)

        ## Thêm

        all_chars = set()

        for word in corpus:
            all_chars.update(word.split())

        for char in sorted(all_chars):
            if char not in vocab:
                vocab[char] = len(vocab)
        
        target_vocab_size = self.vocab_size or float('inf')
        
        ## Kết thêm
        

        # while len(vocab) < (self.vocab_size or float('inf')):
        while len(vocab) < target_vocab_size:
            word_freq = self.get_word_frequency(corpus)

            if not word_freq:
                break

            most_frequent = max(word_freq, key=word_freq.get)

            if word_freq[most_frequent] < self.min_frequency:
                break

            corpus = self.merge_vocab(most_frequent, corpus)

            new_token = ''.join(most_frequent)
            if new_token not in vocab:
                vocab[new_token] = len(vocab)

            self.merges.append(most_frequent)
            
        self.vocab = vocab
        # self.subwords = set(vocab.keys()) - set(self.special_tokens.keys())
        self.subwords = set(vocab.keys())
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def tokenize_word(self, word):
        if not word:
            return []
        
        if word in self.vocab:
            return [word]
        
        chars = ' '.join(list(word)) + ' </w>'

        for pair in self.merges:
            bigram = ' '.join(pair)
            replacement = ''.join(pair)
            chars = chars.replace(bigram, replacement)
        
        word_tokens = chars.split()
        
        tokens = [token if token in self.vocab else '[UNK]' for token in word_tokens]

        return tokens
    
    def tokenize(self, text):
        tokens = []
        for word in text.strip().split():
            tokens.extend(self.tokenize_word(word))
        return tokens
        
    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.special_tokens['[UNK]']) for token in tokens]

    def convert_ids_to_tokens(self, token_ids):
        return [self.id_to_token.get(id, '[UNK]') for id in token_ids]
    
    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_ids = self.convert_tokens_to_ids(tokens) 
        return token_ids
    
    def save_vocab(self, filepath):
        vocab_data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            if isinstance(vocab_data, dict) and 'vocab' not in vocab_data:
                self.vocab = vocab_data
                self.merges = []
            else:
                self.vocab = vocab_data.get('vocab', {})
                self.merges = [tuple(merge) for merge in vocab_data.get('merges', [])]
                self.special_tokens = vocab_data.get('special_tokens', self.special_tokens)
            
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            self.subwords = set(self.vocab.keys())


if __name__ == "__main__":

    data_path = '../data/UIT-VSFC/merge_data/all_text.txt'

    with open(data_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    
    processor = TextPreprocessor()
    texts = [processor.preprocess_text(text) for text in texts]

    tokenizer = BertTokenizer(min_frequency=1)
    tokenizer.build_vocab(texts)

    # print(f"Số lượng token trong vocab: {len(tokenizer.vocab)}")


    # tokenizer.save_vocab('../data/UIT-VSFC/bert_vocab.json')
    # print("Vocab đã được lưu tại: ../data/UIT-VSFC/bert_vocab.json")




    # # Kiểm tra với một câu bất kỳ
    # test_sentence = "Bạn ai đó cả"
    # test_sentence = processor.preprocess_text(test_sentence)
    # tokens = tokenizer.tokenize(test_sentence)
    # token_ids = tokenizer.encode(test_sentence)

    # print("Câu kiểm tra:", test_sentence)
    # print("Tokens:", tokens)
    # print("Token IDs:", token_ids)
    # print("ID to token:", tokenizer.convert_ids_to_tokens(token_ids))




