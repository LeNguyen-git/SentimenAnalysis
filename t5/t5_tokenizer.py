from t5_preprocessing import T5Preprocessing

import json
from collections import defaultdict, Counter
import copy
import math

class T5Tokenizer:
    def __init__(self, vocab_size=None):
        self.vocab_size = vocab_size
        self.freq = {} #Tần xuất
        self.prob = {} #Xác suất
        self.special_tokens = {
            "<pad>",
            "<unk>",
            "<s>",
            "</s>",
        }

        self.token_to_id = {}
        self.id_to_token = {}


        self.preprocessor = T5Preprocessing()

    def get_initial_vocab(self, texts):
        vocab = defaultdict(int)
        for text in texts:
            processed_text = self.preprocessor.preprocess_text(text)
            words = processed_text.split()
            for word in words:
                vocab[word] += 1
        
        character_freqs  = defaultdict(int)
        subwords_freqs  = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word)):
                character_freqs[word[i]] += freq
                for j in range(i + 2, len(word) + 1):
                    subwords_freqs[word[i:j]] += freq
        
        sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
        
        max_tokens = self.vocab_size if self.vocab_size is not None else len(character_freqs) + len(subwords_freqs)
        num_subwords = max(0, max_tokens - len(character_freqs))

        token_freqs = (
            list(character_freqs.items()) + sorted_subwords[:num_subwords]
        )

        token_freqs = {token: freq for token, freq in token_freqs}

        return token_freqs
    
    def viterbi(self, word, prob):
        word_length = len(word)
        
        min_scores = [float('inf')] * (word_length + 1)
        min_scores[0] = 0

        backpointer = [0] * (word_length+1)
        
        for i in range (1, word_length+1):
            for j in range (i):
                subword = word[j:i]
                if subword in prob:
                    score = min_scores[j] + prob[subword]
                    if score < min_scores[i]:
                        min_scores[i] = score
                        backpointer[i] = j

        tokens = []
        i = word_length
        while i > 0:
            j = backpointer[i]
            tokens.append(word[j:i])
            i = j

        return tokens[::-1] if tokens else ["<unk>"], min_scores[word_length]

    # def compute_loss(self, texts, prob):
    #     loss = 0
    #     word_freqs = defaultdict(int)
    #     for text in texts:
    #         words = self.preprocessor.preprocess_text(text).split()
    #         for word in words:
    #             word_freqs[word] += 1
    #         for word, freq in word_freqs.items():
    #             _, word_loss = self.viterbi(word, prob)
    #             loss += freq + word_loss
        
    #     return loss
        

    # def compute_scores(self, prob):
    #     scores = {}
    #     model_loss = self.compute_loss(prob)
    #     for token, scores in prob.items():
    #         if len(token) == 1 or token in self.special_tokens:
    #             continue
    #         prob_without_token = copy.deepcopy(prob)
    #         prob_without_token.pop(token)

    #         scores[token] = self.compute_loss(prob_without_token) - model_loss

    #     return scores

    def tokenize(self, text):
        words = self.preprocessor.preprocess_text(text).split()
        all_tokens = []
        for word in words:
            tokens, _ = self.viterbi(word, self.prob)
            all_tokens.extend(tokens)
        return all_tokens



    def build_vocab(self, texts):
        self.freq = self.get_initial_vocab(texts)
        total_count = sum(self.freq.values())

        self.prob = {token: -math.log(freq/total_count) for token, freq in self.freq.items()}
        self.token_to_id = {token: i for i , token in enumerate(self.special_tokens)}
        next_id = len(self.special_tokens)

        for token in self.freq:
            if token not in self.token_to_id:
                self.token_to_id[token] = next_id
                next_id += 1
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.token_to_id["<s>"])
        token_ids.extend([self.token_to_id.get(t, self.token_to_id["<unk>"]) for t in tokens])
        if add_special_tokens:
            token_ids.append(self.token_to_id["</s>"])
        return token_ids

    def decode(self, token_ids):
        tokens = [self.id_to_token.get(id_, "<unk>") for id_ in token_ids]

        # tokens = [i for i in tokens if i not in {"<s>","</s>","<pad>"}]
        tokens = [i for i in tokens if i != "<pad>"]

        text = " ".join(tokens).strip()

        return text
    
    def save_vocab(self, file_path):
        vocab = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "freq": self.freq,
            "prob": self.prob
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.token_to_id = vocab["token_to_id"]
        self.id_to_token = {int(k): v for k, v in vocab["id_to_token"].items()}
        self.freq = vocab["freq"]
        self.prob = vocab["prob"]
        self.vocab_size = len(self.token_to_id)
    

def main():

    
    data_path = '../data/UIT-VSFC/merge_data/all_text.txt'

    with open(data_path, 'r', encoding='utf-8') as f:
        texts = f.read().splitlines()
    
    processor = T5Preprocessing()
    texts = [processor.preprocess_text(text) for text in texts]

    tokenizer = T5Tokenizer()
    tokenizer.build_vocab(texts)

    # print("-"*80)

    # print(f"Số lượng token trong vocab: {len(tokenizer.token_to_id)}")

    # print("-"*80)

    # tokenizer.save_vocab('../data/UIT-VSFC/t5_vocab.json')
    # print("Vocab đã được lưu tại: ../data/UIT-VSFC/t5_vocab.json")

    print("-"*80)

    # ====== Chạy demo với 1 câu ======
    test_sentence = "Cái khó nha, khó mà làm được á nha"
    print("Câu gốc:", test_sentence)

    tokens = tokenizer.tokenize(test_sentence)
    print("Tokenized:", tokens)

    encoded_ids = tokenizer.encode(test_sentence, add_special_tokens=True)
    print("Token IDs:", encoded_ids)

    subwords = [tokenizer.id_to_token[i] for i in encoded_ids]
    print("Subwords:", subwords)

    decoded_text = tokenizer.decode(encoded_ids)
    print("Decoded:", decoded_text)


if __name__ == "__main__":
    main()

    


        


