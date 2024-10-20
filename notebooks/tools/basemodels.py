import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, random_split, Dataset

import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class Plots:
    def __init__(self, history):
        self.history = history

    def plot_history(self, history=None, dark_mode=False):
        if history is None:
            history = self.history

        if dark_mode:
            plt.style.use('dark_background')
            plt.rcParams['axes.facecolor'] = '#26262e' 
        else:
            plt.style.use('default')

        if not history:
            print("No history to plot")
            return

        losses = history["losses"]

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(losses["train"], label="train_loss")
        if "val" in losses:
            ax[0].plot(losses["val"], label="val_loss")
        ax[0].legend()

        if 'metrics' not in history:
            plt.show()
            return
        metrics = history["metrics"]

        if 'accuracy' in metrics:
            ax[1].plot(metrics["accuracy"], label="val_accuracy")
        if 'precision' in metrics:
            ax[1].plot(metrics["precision"], label="val_precision")
        if 'recall' in metrics:
            ax[1].plot(metrics["recall"], label="val_recall")
        if 'f1' in metrics:
            ax[1].plot(metrics["f1"], label="val_f1")
        ax[1].legend()

        plt.show()

class BaseModel(nn.Module, Plots):
    def __init__(self):
        nn.Module.__init__(self)
        self.history = {}
        Plots.__init__(self, self.history)
        self.compiled = False

    def compile(self, 
                optimizer=None, 
                loss_fn=None,
                device=None,
                metrics=["accuracy"]):

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
        if loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        print(f"Using device: {self.device}")
        self.to(self.device) 

        self.compiled = True
        self.metrics = metrics

    def fit(self, 
            dataset, 
            n_epochs=100, 
            batch_size=64,
            validation_split=0.0, 
            random_seed=42,
            early_stopping=None): # TODO: Implementar early stopping

        if not self.compiled:
            raise Exception("Model not compiled. Please, run model.compile() before fitting.")

        # Determinar o tamanho dos conjuntos de treinamento e validação
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size

        self.history["losses"] = {
            "train": [],
            "val": []
        }
        self.history["metrics"] = {
            'mse': [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

        # Dividir o dataset
        if validation_split > 0.0:
            generator = torch.Generator().manual_seed(random_seed)
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
            print(f"Dataset dividido em {train_size} amostras para treinamento e {val_size} para validação.")
        else:
            train_dataset = dataset
            val_dataset = None
            print("Nenhuma divisão de validação foi realizada. Usando todo o dataset para treinamento.")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        for epoch in range(n_epochs):
            epoch_loss = 0
            batch_loss = 0

            print(f"Epoch {epoch+1}/{n_epochs}")

            # Barra de progresso customizada para o Keras-like TQDM
            progress_bar = tqdm(enumerate(train_loader), 
                                total=len(train_loader), 
                                # desc=f"Epoch {epoch+1}/{n_epochs}", 
                                leave=True)
            for i, batch in progress_bar:
                X, y = batch
                X = X.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self(X)
                loss = self.loss_fn(output, y)
                loss.backward()
                self.optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                progress_bar.set_postfix({"batch_loss": batch_loss})

            epoch_loss /= len(train_loader)

            if val_loader:
                val_loss, val_metrics = self.evaluate(val_loader)
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                self.history["losses"]["train"].append(epoch_loss)
                self.history["losses"]["val"].append(val_loss)
                for k, v in val_metrics.items():
                    self.history["metrics"][k].append(v)
                tqdm.write(f"loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - {metrics_str}")
            else:
                self.history["losses"]["train"].append(epoch_loss)
                tqdm.write(f"Epoch {epoch}/{n_epochs} - loss: {epoch_loss:.4f}")

            print("\n")

        if self.device == "cuda":
            self.cpu()
        return self.history

    def evaluate(self, dataset, batch_size=64, metrics=None):
        if metrics is None and self.metrics is None:
            metrics = ["accuracy"]
        elif metrics is None:
            metrics = self.metrics

        if type(dataset) == DataLoader:
            data_loader = dataset
        else:
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        metric_values = {}

        with torch.no_grad():
            for batch in data_loader:
                X, y = batch
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self(X)

                loss = self.loss_fn(outputs, y)
                total_loss += loss.item()

                # Converter as saídas do modelo para índices de classes preditas
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        avg_loss = total_loss / len(data_loader)

        # Cálculo de métricas
        if "accuracy" in metrics:
            # print(all_labels.shape)
            # print(all_preds.shape)
            accuracy = accuracy_score(all_labels, all_preds)
            metric_values["accuracy"] = accuracy
        if "precision" in metrics:
            precision = precision_score(all_labels, all_preds, zero_division=0)
            metric_values["precision"] = precision
        if "recall" in metrics:
            recall = recall_score(all_labels, all_preds, zero_division=0)
            metric_values["recall"] = recall
        if "f1" in metrics:
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            metric_values["f1"] = f1
        if 'mse' in metrics:
            mse = avg_loss
            metric_values["mse"] = mse
        self.train()
        return avg_loss, metric_values