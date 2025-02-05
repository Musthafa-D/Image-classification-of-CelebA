from networks import CNN
from learner_plot import Learner
from ccbdl.utils import DEVICE
from datetime import datetime
from data_loader import prepare_data


class Normal_run:
    def __init__(self,
                 task,
                 network_config_nlrl: dict,
                 network_config_linear: dict,
                 data_config: dict,
                 learner_config: dict,
                 config,
                 study_path: str,
                 comment: str = "",
                 config_path: str = "",
                 debug: bool = False,
                 logging: bool = False):
        
        self.network_config_nlrl = network_config_nlrl
        self.network_config_linear = network_config_linear
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
        network_nlrl = CNN(3,"Classifier_nlrl", **self.network_config_nlrl).to(DEVICE)
        network_linear = CNN(3,"Classifier_linear", **self.network_config_linear).to(DEVICE)

        self.learner = Learner(self.result_folder,
                               model_nlrl=network_nlrl,
                               model_linear=network_linear,
                               train_data=train_data,
                               test_data=test_data,
                               val_data=val_data,
                               config=self.learner_config,
                               network_config_nlrl=self.network_config_nlrl,
                               network_config_linear=self.network_config_linear,
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
