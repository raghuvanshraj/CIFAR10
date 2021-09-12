class TrainingConfig(object):

    def __init__(
            self,
            batch_size: int,
            lr: int,
    ):
        self.batch_size = batch_size
        self.lr = lr
