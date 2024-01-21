from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from sklearn.metrics import 
import time as T
from IPython import display
from pynput.keyboard import Key, Listener
import torch
import matplotlib.pyplot as plt
import numpy as np
# from torcheval.metrics import MulticlassAccuracy



class DataCollector:
    def __init__(self, batches):
        self.count = 0
        self.losses = [0]
        self.labels = torch.Tensor()
        self.preds = torch.Tensor()
        self.batches = batches


    def batch(self, labels, outputs, loss):
        preds = torch.argmax(outputs, dim=1)
        self.losses[-1] *= self.count
        self.losses[-1] += loss
        self.count += 1
        self.losses[-1] /= self.count
        self.labels = torch.concat((self.labels, labels.cpu()))
        self.preds = torch.concat((self.preds, preds))

        if (self.count % 100 == 0):
            print(self.losses[-1])

    
    def epoch(self):
        self.labels = torch.Tensor()
        self.preds = torch.Tensor()
        self.losses.append(0)
        self.count = 0
    

    def precision_recall_f1(self):
        precision, recall, f1, _ = precision_recall_fscore_support(self.labels, self.preds)
        return precision, recall, f1


    def accuracy(self):
        return accuracy_score(self.labels, self.preds)


    def loss_data(self):
        return self.losses, np.arange(0, len(self.losses) / self.batches, 1 / self.batches)
        


class CallBack:
    def __init__(self, classes: int, train_batches: int, valid_batches: int):
        self.train = DataCollector(train_batches)
        self.valid = DataCollector(valid_batches)
        self.classes = classes
        self.time_point = T.time()


    def train_batch(self, labels, outputs, loss):
        self.train.batch(labels, outputs, loss)
    

    def valid_batch(self, labels, outputs, loss):
        self.valid.batch(labels, outputs, loss)
        

    def epoch(self):
        self.valid.epoch()
        self.train.epoch()


    def _plot_metrics(self, data: DataCollector, index):
        precision, recall, f1 = data.precision_recall_f1()
        self.axes[index].bar(range(self.classes), precision, label='Precision', width=0.3, align='edge')
        self.axes[index].bar(range(self.classes), recall, label='Recall', width=-0.3, align='edge')
        self.axes[index].bar(range(self.classes), f1, label='F1', width=0.3, align='center')
        self.axes[index].set_xlabel('Class')
        self.axes[index].set_ylabel('Metric')
        self.axes[index].legend()
    

    def plot(self,  valid_accuracy=True,
                    train_accuracy=True,
                    time=True,
                    valid_loss=True,
                    train_loss=True,
                    valid_metrics=False,
                    train_metrics=False):
        plt.cla()
        self.train.losses[0] = self.train.losses[1]
        self.valid.losses[0] = self.valid.losses[1]

        plots = 0
        plots = (valid_loss or train_loss) + valid_metrics + train_metrics

        fig, self.axes = plt.subplots(plots, 1, squeeze=False)
        fig.set_size_inches(8, 8 * plots)

        index = 0
        if valid_loss or train_loss:
            if train_loss:
                self.axes[index][0].plot(self.train.losses, label='Training loss')
            if valid_loss:
                self.axes[index][0].plot(self.valid.losses, label='Validation loss')

            self.axes[index][0].legend()
            self.axes[index][0].set_xlabel('Epoch')
            self.axes[index][0].set_ylabel('Loss')
            index += 1
 
        if valid_metrics:
            self._plot_metrics(self.valid, index)
            index += 1
        
        if train_metrics:
            self._plot_metrics(self.train, index)
            index += 1

        display.clear_output(wait=True)
        display.display(fig)
        plt.close(fig)

        if valid_accuracy:
            print("Validation accuracy: ", self.valid.accuracy())

            print(self.valid.labels)
            print(self.valid.preds)
            pass
        
        if train_accuracy:
            print("Train accuracy: ", self.train.accuracy())
            pass

        if time:
            print("Time = ", T.time() - self.time_point)



class Stopper:
    def __init__(self):
        self.stop = False
        self.listen()

    def on_release(self, key):
        if key == Key.esc:
            self.stop = True

    def listen(self):
        self.listener = Listener(
                on_release=self.on_release)
        self.listener.start()
    
    def __call__(self):
        return self.stop
