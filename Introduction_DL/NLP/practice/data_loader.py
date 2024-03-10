from torch.utils.data import DataLoader, Dataset, random_split
import torch
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):
    def __init__(self, src_tensors, trg_tensors):
        self.src_tensors = src_tensors
        self.trg_tensors = trg_tensors

    def __len__(self):
        return len(self.src_tensors)

    def __getitem__(self, idx):
        src_tensor = self.src_tensors[idx]
        trg_tensor = self.trg_tensors[idx]
        return src_tensor, trg_tensor


def create_loaders(train_ratio, batch_size, ru_vocab, en_vocab, data):
    
    ru_sentences, en_sentences = zip(*data)

    ru_tensors = [torch.tensor(ru_vocab.tokenize(sentence), dtype=torch.long) for sentence in ru_sentences]
    en_tensors = [torch.tensor(en_vocab.tokenize(sentence), dtype=torch.long) for sentence in en_sentences]


    # Assuming tensor_dataset is a combined list of (ru_tensor, en_tensor) pairs
    tensor_dataset = list(zip(ru_tensors, en_tensors))

    # Calculate sizes of splits
    train_size = int(len(tensor_dataset) * train_ratio)
    valid_size = len(tensor_dataset) - train_size

    # Split the dataset
    train_dataset, valid_dataset = torch.utils.data.random_split(tensor_dataset, [train_size, valid_size])

    # Wrap with TranslationDataset
    train_dataset = TranslationDataset(*zip(*train_dataset))
    valid_dataset = TranslationDataset(*zip(*valid_dataset))


    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_batch_padded = pad_sequence(src_batch, batch_first=True, padding_value=ru_vocab.special_tokens['<pad>'])
        trg_batch_padded = pad_sequence(trg_batch, batch_first=True, padding_value=en_vocab.special_tokens['<pad>'])
        return src_batch_padded, trg_batch_padded


    # Create DataLoader instances for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return (train_loader, valid_loader)