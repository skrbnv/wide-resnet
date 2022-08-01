class PowerDownSchedulerWithSemiRestarts():
    def __init__(self, optimizer, max_lr=None, min_lr=1e-5, iterations = 100, drop_scale=2) -> None:
        if max_lr is None:
            self.max_lr = optimizer.param_groups[0]['lr']
        else:
            self.max_lr = max_lr
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.iterations = iterations
        self.cur = 0
        self.drop_scale = drop_scale

    def step(self):
        x = self.cur / self.iterations
        lr = max(0, (1-x)**5) * self.max_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(lr, self.min_lr)
        self.cur += 1
        if self.cur == self.iterations:
            self.cur = 0
            self.max_lr /= self.drop_scale


class StepDownScheduler():
    def __init__(self, optimizer, initial_epoch=0) -> None:
        self.multiplier = 0.2
        self.optimizer = optimizer
        self.triggers = [60, 120, 160, 200, 240, 280]
        self.counter = 0
        for self.counter in range(initial_epoch):
            self.step()

    def step(self):
        if self.counter in self.triggers:
            lr = self.optimizer.param_groups[0]['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr*self.multiplier
        self.counter += 1
