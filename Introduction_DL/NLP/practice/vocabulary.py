# batch:
# <sos> привет <eos> <pad> <pad>
# <sos> привет как   дела <eos>


from collections import Counter


class Vocabulary:
    special_tokens = {
        '<pad>': 0,  # Padding token
        '<sos>': 1,  # Start of sentence
        '<eos>': 2,  # End of sentence
        '<unk>': 3   # Unknown word
    }

    def __init__(self, sentences=None, max_size=None):

        self.index_to_token = {}
        self.token_to_index = {}
        
        # Initialize vocabulary with special tokens
        for token, index in self.special_tokens.items():
            self.token_to_index[token] = index
            self.index_to_token[index] = token

        if sentences is None:
            return

        # Count the frequency of each word
        word_freq = Counter(word for sentence in sentences for word in sentence.split())

        # Keep the most common words up to max_size
        most_common_words = word_freq.most_common(max_size-len(self.special_tokens))

        # add most common words to the vocabulary
        for word, _ in most_common_words:
            index = len(self.token_to_index)
            self.token_to_index[word] = index
            self.index_to_token[index] = word


    def tokenize(self, sentence):
        tokenized_seq = [self.special_tokens.get('<sos>')]
        tokenized_seq += [self.token_to_index.get(token, self.special_tokens['<unk>']) for token in sentence.split()]
        tokenized_seq.append(self.special_tokens.get('<eos>'))
        return tokenized_seq


    def detokenize(self, tokens):
        # General detokenization, excluding '<sos>', '<eos>', and '<pad>'
        detokenized = ' '.join(self.index_to_token[token.item()] for token in tokens if token > self.special_tokens['<unk>'])
        return detokenized


    def bleu_detokenize(self, tokens):
        # Detokenize specifically for BLEU calculation, stripping out '<sos>', '<eos>', '<pad>', and '<unk>'
        meaningful_tokens = [token for token in tokens if token > self.special_tokens['<unk>']]
        detokenized = ' '.join(self.index_to_token[token] for token in meaningful_tokens if token not in self.special_tokens.values())
        return detokenized


    def __len__(self):
        return len(self.token_to_index)