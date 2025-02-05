from networks import CNN
from learner import Learner
from ccbdl.utils import DEVICE
from datetime import datetime
from data_loader import prepare_data
from metrics import Metrics
import os
import torch


class Normal_run:
    def __init__(self,
                 task,
                 network_config: dict,
                 data_config: dict,
                 learner_config: dict,
                 config,
                 study_path: str,
                 comment: str = "",
                 config_path: str = "",
                 debug: bool = False,
                 logging: bool = False):
        
        self.network_config = network_config
        self.data_config = data_config
        self.learner_config = learner_config
        self.result_folder = study_path
        self.config = config
        self.task = task
    
    def execute(self):
        start_time = datetime.now()
        
        print("\n\n******* Run is started*******")
        
        # get data
        train_data, test_data, val_data = prepare_data(self.data_config)
        
        # Classifier
        network = CNN(3, "Classifier", **self.network_config).to(DEVICE)
        
        self.learner = Learner(model=network,
                               train_data=train_data,
                               test_data=test_data,
                               val_data=val_data,
                               config=self.learner_config,
                               network_config=self.network_config,
                               result_folder=self.result_folder,
                               logging=True) 
            
        self.learner.fit(test_epoch_step=self.learner_config["testevery"])

        self.learner.parameter_storage.write("Current config:-\n")
        self.learner.parameter_storage.store(self.config)

        self.learner.parameter_storage.write(
            f"Start Time of classifier training and evaluation in this run: {start_time.strftime('%H:%M:%S')}")

        self.learner.parameter_storage.write("\n")

        print("\n\n******* Run is completed*******")

        end_time = datetime.now()

        self.learner.parameter_storage.write(
            f"End Time of classifier training and evaluation in this run: {end_time.strftime('%H:%M:%S')}\n")

        self.duration_trial = end_time - start_time
        self.durations=self.duration_trial

        self.learner.parameter_storage.write(
            f"Duration of classifier training and evaluation in this run: {str(self.durations)[:-7]}\n")

        return self.learner.best_values["TestAcc"]
    
    def eval_metrics(self):
        """
        eval_metrics function of the run
            --> evaluates the metric values for test_data in the run
                and provides final results such as average 
                infidelity and sensitivityof the attributions used.
        Returns
            None.
        """
    
        model_path = os.path.join(self.result_folder, "net_best.pt")
        
        # Classifier
        model = CNN(3, "Classifier", **self.network_config).to(DEVICE)
        
        # Load the model
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        train_data, test_data, val_data = prepare_data(self.data_config)
        test_metrics = Metrics(model=model, test_data=test_data, result_folder=self.result_folder,
                                best_trial_check=1)
        
        test_metrics.calculations()
        duration_metrics_run = test_metrics.total_metric_duration()
        duration_per_run = self.durations + duration_metrics_run

        with open(os.path.join(self.result_folder, "ParameterStorage.txt"), "a") as file:
            file.write(f"Duration of metrics calculation of test data in this run: {str(duration_metrics_run)[:-7]}\n\n")
            file.write(f"Total duration of this run: {str(duration_per_run)[:-7]}")
