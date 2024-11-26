from code.scheduler.base_scheduler import MultiStepLRScheduler,CosineAnnealingLRScheduler,ReduceLROnPlateauScheduler


class SchedulerSelector():
    """
    scheduler를 새롭게 추가하기 위한 방법
        1. torch에서 제공하는 scheduler는 scheduler_selector.py에 함수로 구현
        2. 직접 구현해야하는 scheduler는 scheduler 폴더 내부에 class로 구현
        2. 구현한 Scheduler Class를 scheduler_selector.py 내부로 import
        3. self.scheduler_classes에 아래와 같은 형식으로 추가
        4. yaml파일의 scheduler_name을 설정한 key값으로 변경
    """
    def __init__(self) -> None:
        self.scheduler_classes = {
            "MultiStepLR" : MultiStepLRScheduler,
            "CosineAnnealingLR" : CosineAnnealingLRScheduler,
            "ReduceLROnPlateau": ReduceLROnPlateauScheduler
        }
    

    def get_scheduler(self, scheduler_name, **scheduler_parameter):
        scheduler_class = self.scheduler_classes.get(scheduler_name)
        if scheduler_class is None:
            raise ValueError(f"Scheduler '{scheduler_name}' not found. Available schedulers: {list(self.scheduler_classes.keys())}")
        return scheduler_class(**scheduler_parameter)



