import torch
import torch.nn as nn
import numpy as np

from vocabulary import Vocabulary


class Seq2SeqLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            embedding_decoder_dim,
            embedding_encoder_dim,
            hidden_decoder_dim,
            hidden_encoder_dim,
            num_decoder_layers,
            num_encoder_layers,
            dropout_rate
):
        
        super(Seq2SeqLSTM, self).__init__()
        
        # Define the embedding layer for the source language with input_size and embedding_dim
        self.encoder_embedding = nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=embedding_encoder_dim
        )
        
        # Define the LSTM encoder with embedding_dim, hidden_dim, num_layers
        self.encoder_lstm = nn.LSTM(
            input_size=embedding_encoder_dim,
            hidden_size=hidden_encoder_dim,
            num_layers=num_encoder_layers,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Define the embedding layer for the decoder
        self.decoder_embedding = nn.Embedding(
            num_embeddings=output_size,
            embedding_dim=embedding_decoder_dim
        )
        
        # Define the LSTM decoder with embedding_dim, hidden_dim, num_layers
        self.decoder_lstm = nn.LSTM(
            input_size=embedding_decoder_dim,
            hidden_size=hidden_decoder_dim,
            num_layers=num_decoder_layers,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Define the fully connected layer to map the decoder outputs to the target vocabulary size
        self.fc_out = nn.Linear(
            in_features=hidden_decoder_dim,
            out_features=output_size
        )

        # Define the dropout regularization
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
        output_size = self.fc_out.out_features

        # Initialize the output tensor to store the decoder's predictions
        outputs = torch.zeros(batch_size, target_len, output_size).to(input_seq.device)

        # Embed the input sequence
        encoder_embedded = self.dropout(self.encoder_embedding(input_seq))

        # Encoder
        _, (hidden, cell) = self.encoder_lstm(encoder_embedded)
        
        # Prepare the first input to the decoder, which is <sos> tokens
        decoder_input = torch.tensor([[Vocabulary.special_tokens['<sos>']] for _ in range(batch_size)], device=input_seq.device)

        # Decoder
        for t in range(1, target_len):
            decoder_embedded = self.dropout(self.decoder_embedding(decoder_input))
            decoder_output, (hidden, cell) = self.decoder_lstm(decoder_embedded, (hidden, cell))

            output = self.fc_out(decoder_output.squeeze(1))
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            decoder_input = (target_seq[:, t] if teacher_force else top1).unsqueeze(1)
            
        return outputs


    def _forward_infer(self, input_seq, max_length):
        batch_size = input_seq.size(0)
        output_size = self.fc_out.out_features
        outputs = torch.zeros(batch_size, max_length, output_size).to(input_seq.device)
        
        # Encoder
        encoder_embedded = self.dropout(self.encoder_embedding(input_seq))
        _, (hidden, cell) = self.encoder_lstm(encoder_embedded)

        # Decoder
        decoder_input = torch.tensor([Vocabulary.special_tokens['<sos>']] * batch_size).to(input_seq.device)
        for t in range(1, max_length):
            decoder_embedded = self.dropout(self.decoder_embedding(decoder_input).unsqueeze(1))
            decoder_output, (hidden, cell) = self.decoder_lstm(decoder_embedded, (hidden, cell))
            output = self.fc_out(decoder_output.squeeze(1))
            outputs[:, t, :] = output
            
            # Get the highest probability token
            top1 = output.max(1)[1]
            decoder_input = top1
            
            # Stop at EOS tokens (you'll need to define EOS_token)
            if (top1 == Vocabulary.special_tokens['<eos>']).all():
                break
                
        return outputs
    

def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight_ih' in name:  # LSTM input-hidden weights
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:  # LSTM hidden-hidden weights
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:  # Biases
            param.data.fill_(0)
        elif 'weight' in name:  # Linear layer weights
            nn.init.xavier_uniform_(param.data)
        elif 'embedding' in name:  # Embedding weights
            nn.init.uniform_(param.data, -0.1, 0.1)