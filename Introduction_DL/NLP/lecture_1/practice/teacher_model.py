import torch.nn as nn
import torch

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
        super(Seq2SeqLSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout_rate, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, input_seq, target_seq=None, teacher_forcing_ratio=0.5, max_length=50):
        if self.training:
            # Training mode: target_seq is required for teacher forcing
            return self._forward_train(input_seq, target_seq, teacher_forcing_ratio)
        else:
            # Inference mode: generate output sequence without target_seq
            return self._forward_infer(input_seq, max_length)
    
    def _forward_train(self, input_seq, target_seq, teacher_forcing_ratio):
        batch_size, target_len = target_seq.size()
        output_size = self.fc.out_features
        
        # Initialize the output tensor to store the decoder's predictions
        outputs = torch.zeros(batch_size, target_len, output_size).to(input_seq.device)
        
        # Embed the input sequence
        embedded_input = self.dropout(self.embedding(input_seq))
        
        # Encoder
        _, (hidden, cell) = self.encoder_lstm(embedded_input)
        
        # Prepare the first input to the decoder, which is <sos> tokens
        decoder_input = torch.tensor([[Vocabulary.special_tokens['<sos>']] for _ in range(batch_size)], device=input_seq.device)
        
        # Decoder: Iterate through the sequence
        for t in range(0, target_len):
            embedded = self.dropout(self.embedding(decoder_input))
            # One step of the decoder
            decoder_output, (hidden, cell) = self.decoder_lstm(embedded, (hidden, cell))
            # Predict next token
            output = self.fc(decoder_output.squeeze(1))
            outputs[:, t, :] = output
            # Determine if we will use teacher forcing or not
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            # Get the highest probability token
            top1 = output.argmax(1)
            # If teacher forcing, use actual next token as next input. If not, use predicted token
            decoder_input = (target_seq[:, t] if teacher_force else top1).unsqueeze(1)
        
        return outputs
    
    # tar: ... the dog
    # gen: ... the cat

    def _forward_infer(self, input_seq, max_length=50):
        batch_size = input_seq.size(0)
        output_size = self.fc.out_features
        outputs = torch.zeros(batch_size, max_length, output_size).to(input_seq.device)
        
        # Encoder
        embedded_input = self.dropout(self.embedding(input_seq))
        _, (hidden, cell) = self.encoder_lstm(embedded_input)
        
        # Start token (assuming it's the first index)
        decoder_input = torch.tensor([[Vocabulary.special_tokens['<sos>']] for _ in range(batch_size)], device=input_seq.device)
        
        for t in range(1, max_length):
            embedded = self.dropout(self.embedding(decoder_input))
            decoder_output, (hidden, cell) = self.decoder_lstm(embedded, (hidden, cell))
            output = self.fc(decoder_output.squeeze(1))
            outputs[:, t, :] = output
            
            top1 = output.argmax(1)
            decoder_input = top1.unsqueeze(1)
            if torch.all(top1 == Vocabulary.special_tokens['<eos>']):
                break
        
        return outputs
