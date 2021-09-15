class MetricsRecorder(object):

    def __init__(self, name: str):
        self.name = name
        self.update_count = 0.
        self.running_loss = 0.
        self.running_accuracy = 0.
        self.losses = []
        self.accuracies = []

    def update(self, loss: float, accuracy: float):
        self.running_loss += loss
        self.running_accuracy += accuracy
        self.update_count += 1

    def clear(self):
        self.update_count = 0.
        self.running_loss = 0.
        self.running_accuracy = 0.

    def record(self):
        self.losses.append(self.running_loss / self.update_count)
        self.accuracies.append(self.running_accuracy / self.update_count)
        self.clear()

    def print(self, epoch: int, color_prefix: str):
        data = [f'{self.name}:', f'epoch: {epoch}', f'loss: {self.losses[-1]}', f'accuracy: {self.accuracies[-1]}']
        print(color_prefix + '{: <20}{: <20}{: <30}{: <30}'.format(*data))
