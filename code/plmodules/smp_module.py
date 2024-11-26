import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Dice, JaccardIndex  # Dice와 IoU 사용
from code.models.model_selector import ModelSelector  # ModelSelector 사용
from code.loss_functions.loss_selector import LossSelector  # LossSelector 사용
from code.scheduler.scheduler_selector import SchedulerSelector  # SchedulerSelector 사용
from code.optimizers.optimizer_selector import OptimizerSelector
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR  # 고정된 스케줄러 사용
class XrayModelModule(pl.LightningModule):
    def __init__(self, config):
        super(XrayModelModule, self).__init__( )
        self.config = config

        # ModelSelector로 모델 초기화
        model_selector = ModelSelector()
        self.model = model_selector.get_model(
            model_name=config.model.model_name,
            **config.model.parameters
        )
        if not self.model:
            raise ValueError("ModelSelector failed to initialize the model.")
        # print("model_selctor completed!") #디버깅용
       

        # LossSelector로 Loss 함수 초기화
        loss_selector = LossSelector()
        self.criterion = loss_selector.get_loss(
            config.loss.name,
            **config.loss.parameters
        )
        if not self.criterion:
            raise ValueError("LossSelector failed to initialize the loss function.")
        
        # print(f"Initialized loss: {config.loss.name}") #디버깅용

               
         # optimizer는 고정
        self.optimizer = optim.Adam(params=self.model.parameters(),
                            lr=self.config.optimizer.parameter.lr,
                            weight_decay=self.config.optimizer.parameter.weight_decay)
            
       # Scheduler 고정
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # # WandB 설정 -> Wandb 로깅 선언은 보통 pl.Trainer 에서 설정되기 떄문에 LightningModule 내부에 포함되지 않아도 됨.
        # self.wandb_logger = WandbLogger(project=config.wandb.project, name=config.wandb.name)

    def forward(self, x):
        return self.model(x)

    # def configure_optimizers(self):
    #     if self.scheduler:
    #         return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
    #     return self.optimizer
