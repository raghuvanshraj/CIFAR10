from typing import Callable, Iterator, Mapping, Any
from typing import Optional

import torch
import torch.cuda
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from colorama import Fore
from torch.nn.parameter import Parameter

from src.training.config import TrainingConfig
from src.training.metrics import MetricsRecorder


class ModelTrainer(object):

    def __init__(
            self,
            model: nn.Module,
            optimizer: Callable[[Iterator[Parameter], int], torch.optim.Optimizer],
            criterion: nn.Module,
            training_config: TrainingConfig,
            training_data: data.Dataset,
            cuda: Optional[bool] = False,
            validation_data: Optional[data.Dataset] = None,
    ):
        self.train_loader = data.DataLoader(training_data, batch_size=training_config.batch_size, shuffle=True)
        self.validation_loader = data.DataLoader(validation_data, batch_size=training_config.batch_size, shuffle=True)
        self.criterion = criterion

        self.device = torch.device('cpu')
        if cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
                print(f'using {torch.cuda.get_device_name(self.device.index)}')
            else:
                print('no gpu available, processing on cpu')

        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), training_config.lr)

    def train(self, epochs: int, validation: bool) -> Iterator[MetricsRecorder]:
        training_metrics = MetricsRecorder('training metrics')
        validation_metrics = MetricsRecorder('testing metrics')

        if validation and self.validation_loader is None:
            print('validation data not provided, setting validation to false')
            validation = False

        for e in range(epochs):
            for x, y in self.train_loader:
                loss, accuracy = self.single_pass(x, y)
                training_metrics.update(loss, accuracy)

            training_metrics.record()
            training_metrics.print(e, Fore.CYAN)

            if validation:
                with torch.no_grad():
                    for x, y in self.validation_loader:
                        loss, accuracy = self.single_pass(x, y)
                        validation_metrics.update(loss, accuracy)

                    validation_metrics.record()
                    validation_metrics.print(e, Fore.YELLOW)

        return training_metrics, validation_metrics

    def single_pass(self, x: torch.Tensor, y: torch.Tensor) -> [float, float]:
        x = x.to(self.device)
        y = y.to(self.device)

        y_pred: torch.Tensor = self.model(x)
        loss = self.criterion(y_pred, y)

        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        preds = torch.argmax(y_pred, dim=1)
        return loss.item(), (torch.sum(preds == y.data).item() / y.shape[0]) * 100

    def predict(
            self,
            img: Image,
            transform: Callable,
            preprocess: Optional[Callable] = None,
            classes: Optional[Mapping[int, str]] = None
    ) -> Any:
        if preprocess:
            img = preprocess(img)

        img = transform(img)
        img = img.to(self.device).unsqueeze(0)
        with torch.no_grad():
            y_pred = self.model(img)

        pred = torch.argmax(y_pred, dim=1)
        return classes[pred.item()] if classes else pred.item()
