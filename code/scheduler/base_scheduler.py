import torch
from torch.optim import lr_scheduler

class MultiStepLRScheduler:
    """
    MultiStepLR Scheduler
    """
    def __init__(self, scheduler, **scheduler_parameters):
        """
        Initializes the MultiStepLR scheduler.

        :param scheduler: Base scheduler to apply the learning rate policy on.
        :param scheduler_parameters: Additional parameters for MultiStepLR (milestones, gamma, etc.)
        """
        self.scheduler = lr_scheduler.MultiStepLR(scheduler, **scheduler_parameters)

    def step(self):
        """
        Advances the scheduler by one step.
        """
        self.scheduler.step()

    def get_lr(self):
        """
        Retrieves the current learning rates.
        """
        return self.scheduler.get_last_lr()


class CosineAnnealingLRScheduler:
    """
    CosineAnnealingLR Scheduler
    """
    def __init__(self, scheduler, **scheduler_parameters):
        """
        Initializes the CosineAnnealingLR scheduler.

        :param scheduler: Base scheduler to apply the learning rate policy on.
        :param scheduler_parameters: Additional parameters for CosineAnnealingLR (T_max, eta_min, etc.)
        """
        self.scheduler = lr_scheduler.CosineAnnealingLR(scheduler, **scheduler_parameters)

    def step(self):
        """
        Advances the scheduler by one step.
        """
        self.scheduler.step()

    def get_lr(self):
        """
        Retrieves the current learning rates.
        """
        return self.scheduler.get_last_lr()


class ReduceLROnPlateauScheduler:
    """
    ReduceLROnPlateau Scheduler
    """
    def __init__(self, scheduler, **scheduler_parameters):
        """
        Initializes the ReduceLROnPlateau scheduler.

        :param scheduler: Base scheduler to apply the learning rate policy on.
        :param scheduler_parameters: Additional parameters for ReduceLROnPlateau 
                                      (mode, factor, patience, threshold, etc.)
        """
        self.scheduler = lr_scheduler.ReduceLROnPlateau(scheduler, **scheduler_parameters)

    def step(self, metrics):
        """
        Advances the scheduler based on the specified metrics.

        :param metrics: The value of the monitored metric to decide on the learning rate adjustment.
        """
        self.scheduler.step(metrics)

    def get_lr(self):
        """
        Retrieves the current learning rates.
        Note: ReduceLROnPlateau does not expose a direct get_last_lr method.
        """
        for param_group in self.scheduler.optimizer.param_groups:
            return param_group['lr']
