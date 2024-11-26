from .base_optimizer import adam_optimizer, sgd_optimizer, rmsprop_optimizer


class OptimizerSelector():
    """
    optimizer를 새롭게 추가하기 위한 방법
        1. torch에서 제공하는 optimizer는 optimizer_selector.py에 함수로 구현
        2. 직접 구현해야 하는 optimizer는 base_optimizer.py 내부에 추가
        3. 구현한 Optimizer Class를 optimizer_selector.py 내부로 import
        4. self.optimizer_classes에 아래와 같은 형식으로 추가
        5. yaml파일의 optimizer_name을 설정한 key값으로 변경
    """
    def __init__(self, model_params) -> None:
        self.optimizer_classes = {
            "Adam": adam_optimizer,
            "SGD": sgd_optimizer,
            "RMSprop": rmsprop_optimizer
        }
        self.model_params = model_params

    def get_optimizer(self, optimizer_name, **optimizer_params):
        if optimizer_name not in self.optimizer_classes:
            raise ValueError(f"Optimizer '{optimizer_name}' is not implemented.")
        return self.optimizer_classes[optimizer_name](self.model_params, **optimizer_params)
