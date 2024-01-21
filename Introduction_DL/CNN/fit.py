import callback as cb
import torch


def fit(model, train_dataset, valid_dataset,
        criterion, optimizer, epochs, classes,
        backup_path: str | None = None, stopper: bool = False):
    callback = cb.CallBack(classes, len(train_dataset), len(valid_dataset))


    if stopper:
        stop = cb.Stopper()

    for epoch in range(epochs):  # loop over the dataset multiple times
        callback.epoch()

        model.train()  # Set the model back to training mode
        for i, data in enumerate(train_dataset):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            callback.train_batch(labels, outputs, loss.item())

            if stopper and stop():
                break
        
        if stopper and stop():
                break

        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode
            for data in valid_dataset:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                callback.valid_batch(labels, outputs, loss.item())

        if backup_path:
            torch.save(model.state_dict(), f'{backup_path}_epoch_{epoch}')
    
        callback.plot()
    
    print('Finished Training')
