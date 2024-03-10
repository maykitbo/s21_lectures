import torch
import torch.nn as nn
from torch import optim
import progressbar
from nltk.translate.bleu_score import corpus_bleu

from vocabulary import Vocabulary


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead.")


def train_model(
        model,
        train_loader,
        valid_loader,
        target_vocab,
        num_epochs=10,
        learning_rate=0.001,
        weight_decay=0.0,
        teacher_forcing_ratio=0.0,
        name='v1',
        scores_to_file=True
):
    
    train_losses = []
    train_epoch_losses = []
    valid_losses = []
    valid_epoch_losses = []
    bleu_scores = []

    # Use CrossEntropyLoss for classification tasks
    criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.special_tokens['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    for epoch in range(num_epochs):

        # Train
        losses, avg_train_loss = _one_epoch_train(model ,train_loader, optimizer, teacher_forcing_ratio, criterion)
        train_epoch_losses.append(avg_train_loss)
        train_losses += losses

        # Validation
        losses, avg_val_loss, bleu_score = _one_epoch_validate(model, valid_loader, criterion, target_vocab)
        valid_epoch_losses.append(avg_val_loss)
        valid_losses += losses
        bleu_scores.append(bleu_score)

        torch.save(model.state_dict(), f'backup/model_{name}_epoch_{epoch}')

        print(f'''Epoch {epoch+1}:
              Train Loss: {avg_train_loss:.4f},
              Validation Loss: {avg_val_loss:.4f},
              BLUE score: {bleu_score:.4f}''')

    print("Training complete.")

    if scores_to_file:
        with open(f'backup/model_{name}_epoch_{epoch}.txt', 'a') as f:
            f.write(f'train_epoch_losses: {train_epoch_losses}\n\n')
            f.write(f'valid_epoch_losses: {valid_epoch_losses}\n\n')
            f.write(f'bleu_scores: {bleu_scores}\n\n')




def _one_epoch_train(model, train_loader, optimizer, teacher_forcing_ratio, criterion):
    train_losses = []

    model.train()  # Set model to training mode
    total_loss = 0

    bar = progressbar.ProgressBar(maxval=len(train_loader)).start()
    bar_idx = 0

    for input_tensors, target_tensors in train_loader:

        input_tensors = input_tensors.to(device)
        target_tensors = target_tensors.to(device)

        optimizer.zero_grad()  # Clear gradients

        output = model(
            input_seq=input_tensors,
            target_seq=target_tensors,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        # Compute loss; assuming output shape is [batch_size, seq_len, output_size]
        # and target shape is [batch_size, seq_len]
        loss = criterion(output.view(-1, output.size(-1)), target_tensors.view(-1))
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        total_loss += loss.item()

        train_losses.append(loss.item())

        bar.update(bar_idx)
        bar_idx += 1

    avg_train_loss = total_loss / len(train_loader)

    return (train_losses, avg_train_loss)
    # train_epoch_losses.append(avg_train_loss)



def _one_epoch_validate(model, valid_loader, criterion, target_vocab):
    # Validation loop
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    references = []
    hypotheses = []
    valid_losses = []

    with torch.no_grad():
        for input_tensors, target_tensors in valid_loader:

            input_tensors = input_tensors.to(device)
            target_tensors = target_tensors.to(device)

            output = model(input_tensors, max_length=target_tensors.shape[1])
            loss = criterion(output.view(-1, output.size(-1)), target_tensors.view(-1))
            total_val_loss += loss.item()
            valid_losses.append(loss.item())

            # Convert model output and target tensors to word indices
            # Assuming output is [batch_size, seq_len, output_size]
            output_indices = output.argmax(2)  # Get the index of the max log-probability
            
            for i in range(output_indices.size(0)):
                # Convert target tensor indices to words, excluding special tokens
                ref = target_vocab.bleu_detokenize(target_tensors[i].tolist())
                # Convert hypothesis (model output) indices to words, excluding special tokens
                hyp = target_vocab.bleu_detokenize(output_indices[i].tolist())
                references.append([ref.split()])  # BLEU expects tokenized sentences as a list of words
                hypotheses.append(hyp.split())

    avg_val_loss = total_val_loss / len(valid_loader)
    # valid_epoch_losses.append(avg_val_loss)
    bleu_score = corpus_bleu(references, hypotheses)

    return (valid_losses, avg_val_loss, bleu_score)


