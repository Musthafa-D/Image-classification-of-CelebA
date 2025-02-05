import os
import torch
from metrics import Metrics
from ccbdl.utils import DEVICE
from data_loader import prepare_data
from networks import CNN
from ccbdl.config_loader.loaders import ConfigurationLoader


config_path = os.path.join(os.getcwd(), "dummy_config.yaml")
config = ConfigurationLoader().read_config("dummy_config.yaml")

network_config = config["network"]
data_config = config["data"]

result_path = os.path.join(os.getcwd())

model = CNN(3,"Classifier", **network_config).to(DEVICE)

# Load the model
if network_config["final_layer"] == "nlrl":
    model_path = os.path.join(result_path, "net_best_nlrl.pt")
    checkpoint = torch.load("net_best_nlrl.pt")
else:
    model_path = os.path.join(result_path, "net_best_linear.pt")
    checkpoint = torch.load("net_best_linear.pt")
    
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

train_data, test_data, val_data = prepare_data(data_config)

# Compute the metrics
test_metrics = Metrics(model=model, test_data=test_data, result_folder=result_path,
                       best_trial_check=1)

test_metrics.calculations()