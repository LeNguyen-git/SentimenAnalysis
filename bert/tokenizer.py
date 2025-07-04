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

        for word, freq in corpus.items():
            # symbols = re.findall(r'\w+', words)
            symbols = word.split()
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

    # def tokenizer_word(self, word):
    #     if not word:
    #         return []
        
    #     if word in self.vocab:
    #         return [word]
        
    #     chars = ' '.join(list(word)) + ' </w>'

    #     for pair in self.merges:
    #         bigram = ' '.join(pair)
    #         replacement = ''.join(pair)
    #         chars = chars.replace(bigram, replacement)
        
    #     word_tokens = chars.split()
        
    #     tokens = [token if token in self.vocab else '[UNK]' for token in word_tokens]

    #     return tokens

    def tokenizer_word(self, word):
        if not word:
            return []

        if word in self.vocab:
            return [word]

        symbols = list(word)
        if symbols:
            symbols[-1] = symbols[-1] + '</w>' 

        merges = list(symbols)

        while True:
            pairs = [(merges[i], merges[i + 1]) for i in range(len(merges) - 1)]
            if not pairs:
                break

            merge_candidate = None
            for pair in self.merges:
                if pair in pairs:
                    merge_candidate = pair
                    break

            if not merge_candidate:
                break

            new_merges = []
            i = 0
            while i < len(merges):
                if i < len(merges) - 1 and (merges[i], merges[i + 1]) == merge_candidate:
                    new_merges.append(merges[i] + merges[i + 1])
                    i += 2
                else:
                    new_merges.append(merges[i])
                    i += 1
            merges = new_merges

        tokens = []
        for token in merges:
            if token in self.vocab:
                tokens.append(token)
            else:
                for ch in token:
                    tokens.append(ch if ch in self.vocab else '[UNK]')
        return tokens


    
    def tokenizer(self, text):
        tokens = []
        for word in text.strip().split():
            tokens.extend(self.tokenizer_word(word))
        return tokens
        
    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.special_tokens['[UNK]']) for token in tokens]

    def convert_ids_to_tokens(self, token_ids):
        return [self.id_to_token.get(id, '[UNK]') for id in token_ids]
    
    def encode(self, text_a, text_b=None, add_special_tokens=True):
        tokens_a = self.tokenizer(text_a)
        tokens_b = self.tokenizer(text_b) if text_b is not None else []

        if add_special_tokens:
            tokens = ['[CLS]'] + tokens_a + ['[SEP]']
            if tokens_b:
                tokens += tokens_b + ['[SEP]']
        else:
            tokens = tokens_a + (['[SEP]'] if tokens_b else [])
        
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids
    
    def token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get('[UNK]', 1))

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
            self.vocab_size = len(self.vocab)


if __name__ == "__main__":

    data_path = '../data/UIT-VSFC/merge_data/all_text.txt'

    with open(data_path, 'r', encoding='utf-8') as f:
        texts = f.read().splitlines()
    
    processor = TextPreprocessor()
    texts = [processor.preprocess_text(text) for text in texts]

    tokenizer = BertTokenizer(min_frequency=1)
    tokenizer.build_vocab(texts)

    # print(f"Số lượng token trong vocab: {len(tokenizer.vocab)}")

    # tokenizer.save_vocab('../data/UIT-VSFC/bert_vocab.json')
    # print("Vocab đã được lưu tại: ../data/UIT-VSFC/bert_vocab.json")


    # # Kiểm tra với một câu bất kỳ
    # test_sentence = "Chào bạn hôm nay trời đẹp quá!. Có ai ở đó không ?"
    # test_sentence = processor.preprocess_text(test_sentence)
    # tokens = tokenizer.tokenizer(test_sentence)
    # token_ids = tokenizer.encode(test_sentence, add_special_tokens=True)

    # print("Câu kiểm tra:", test_sentence)
    # print("Tokens:", tokens)
    # print("Token IDs:", token_ids)
    # print("ID to token:", tokenizer.convert_ids_to_tokens(token_ids))

    # # Kiểm tra với 2 câu
    # sentence_a = "Tôi thích đọc sách vào buổi sáng."
    # sentence_b = "Đó là cách tôi bắt đầu một ngày mới."

    # # Tiền xử lý 2 câu
    # sentence_a = processor.preprocess_text(sentence_a)
    # sentence_b = processor.preprocess_text(sentence_b)

    # # Token hóa từng câu riêng biệt (để xem từng bước)
    # tokens_a = tokenizer.tokenizer(sentence_a)
    # tokens_b = tokenizer.tokenizer(sentence_b)

    # # Mã hóa 2 câu cùng lúc (BERT-style)
    # token_ids = tokenizer.encode(sentence_a, sentence_b, add_special_tokens=True)

    # sep_index = token_ids.index(tokenizer.token_to_id("[SEP]"))
    # token_type_ids = [0] * (sep_index + 1) + [1] * (len(token_ids) - (sep_index + 1))

    # # In kết quả
    # print("Câu A:", sentence_a)
    # print("Câu B:", sentence_b)
    # print("Tokens A:", tokens_a)
    # print("Tokens B:", tokens_b)
    # print("Token IDs:", token_ids)
    # print("ID to Token:", tokenizer.convert_ids_to_tokens(token_ids))
    # print("Token Type IDs:", token_type_ids)





