import torch
import numpy as np
import os
import time
from captum.metrics import infidelity, sensitivity_max
from tabulate import tabulate
from ccbdl.utils import DEVICE
from datetime import datetime
from captum.attr import Saliency, GuidedBackprop, InputXGradient, Deconvolution, Occlusion

class Metrics:
    def __init__(self, model, test_data, result_folder, best_trial_check):
        self.model = model
        self.device = DEVICE
        self.test_data = test_data
        self.result_folder = result_folder
        self.best_trial_check = best_trial_check
        self.total_duration = 0
    
    def compute_metrics(self, method_name, method, method_map, inputs, labels, label_index):
        def my_perturb_func(inputs):
            noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float().to(self.device)
            return noise, inputs - noise

        target_index = torch.full((inputs.size(0),), label_index, dtype=torch.int64).to(self.device)
        infidelity_score = infidelity(self.model, my_perturb_func, inputs, method_map, target=target_index)
        
        if method_name == "Occlusion":
            sensitivity_score = sensitivity_max(method.attribute, inputs, target=target_index, sliding_window_shapes=(3, 10, 10), 
                                     baselines=torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(DEVICE),
                                     strides=(3, 5, 5))
        else:
            sensitivity_score = sensitivity_max(method.attribute, inputs, target=target_index)
        return infidelity_score, sensitivity_score

    def calculations(self):
        start_time = datetime.now()

        num_labels = 10  # Number of labels in CelebA dataset that were chosen
        if self.best_trial_check == 1:
            method_names = ["Saliency", "Guided Backprop", "Input X Gradient", "Deconvolution", "Occlusion"]
            include_occlusion = 1
        else:
            method_names = ["Saliency", "Guided Backprop", "Input X Gradient", "Deconvolution"]
            include_occlusion = 0
            
        metrics_data = {method: {"infidelity": [0.0] * num_labels, "sensitivity": [0.0] * num_labels} for method in method_names}
        method_durations = {method: [0.0] * num_labels for method in method_names}
        
        total_samples = 0

        for i, data in enumerate(self.test_data):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device).long()
            inputs.requires_grad = True

            # Get attribution methods and maps
            attr_methods = attributions(self.model, include_occlusion)
            attr_maps_dict = attribution_maps(self.model, inputs, labels, include_occlusion)

            # Compute metrics for each label
            for label_index in range(num_labels):
                for method_name, method in zip(method_names, attr_methods):
                    method_start_time = time.time()
                    
                    attr_map = attr_maps_dict[method_name][label_index]
                    
                    infid, sens = self.compute_metrics(method_name, method, attr_map, inputs, labels, label_index)
                    
                    metrics_data[method_name]["infidelity"][label_index] += infid.sum().item()
                    metrics_data[method_name]["sensitivity"][label_index] += sens.sum().item()
                    
                    method_end_time = time.time()
                    duration = method_end_time - method_start_time
                    method_durations[method_name][label_index] += duration
                    
                    print(f"{method_name}: {method_durations[method_name][label_index]} for label {label_index}")
            
            print(f"{i}\n")
            total_samples += inputs.size(0)

        # Calculate average scores per label and overall
        for method in method_names:
            for label_index in range(num_labels):
                metrics_data[method]["infidelity"][label_index] /= total_samples
                metrics_data[method]["sensitivity"][label_index] /= total_samples
            metrics_data[method]["overall_infidelity"] = sum(metrics_data[method]["infidelity"]) / num_labels
            metrics_data[method]["overall_sensitivity"] = sum(metrics_data[method]["sensitivity"]) / num_labels

        end_time = datetime.now()
        self.total_duration = end_time - start_time
        
        # Save metrics to file
        self.save_metrics_to_file(metrics_data)

    def save_metrics_to_file(self, metrics_data):
        output_path = os.path.join(self.result_folder, "metric_values_of_test_dataset")
        os.makedirs(output_path, exist_ok=True)

        table_data = []
        label_names = ["Bald", "Bangs", "Black_hair", "Eyeglasses", "Male", "Mustache", "No_Beard", "Smiling", "Wearing_Hat", "Young"]
        for method, data in metrics_data.items():
            for label_index, label_name in enumerate(label_names):
                row = [f"{method} - {label_name}", data["infidelity"][label_index], data["sensitivity"][label_index]]
                table_data.append(row)
            # Add overall metrics
            row = [f"{method} - Overall", data["overall_infidelity"], data["overall_sensitivity"]]
            table_data.append(row)

        table_headers = ["Attribution Method", "Average Infidelity", "Average Sensitivity"]
        table_string = tabulate(table_data, headers=table_headers, tablefmt="grid")
        
        with open(os.path.join(output_path, "metrics.txt"), "w") as file:
            file.write("Metrics of CelebA Test Dataset\n\n")
            file.write(table_string)
            file.write("\n\n")
            file.write(f"\nTotal duration for calculating metrics: {self.format_duration(self.total_duration)}\n")
            file.write("Duration for Calculating Metrics of CelebA Test Dataset\n\n")
            for method, durations in self.method_durations.items():
                total_duration = sum(durations)
                file.write(f"Total duration for {method}: {self.format_duration(total_duration)}\n")
                for label_index, duration in enumerate(durations):
                    label_name = label_names[label_index]
                    file.write(f"Duration for {method} - {label_name}: {self.format_duration(duration)}\n")
                file.write("\n")

    def format_duration(self, duration):
        """Converts seconds to a string in the format hours:minutes:seconds."""
        h, remainder = divmod(duration, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)} hours, {int(m)} minutes, {s:.2f} seconds"


    def total_metric_duration(self):
        return self.total_duration
    
    
def attributions(model, include_occlusion):
    # Initialize the Saliency object
    saliency = Saliency(model)
    # Initialize the GuidedBackprop object
    guided_backprop = GuidedBackprop(model)
    # Initialize the DeepLift object
    input_x_gradient = InputXGradient(model)
    # Initialize the Deconvolution object
    deconv = Deconvolution(model)
    
    if include_occlusion == 1:
        # Initialize the Occlusion object
        occlusion = Occlusion(model)
        return saliency, guided_backprop, input_x_gradient, deconv, occlusion
    
    else:
        return saliency, guided_backprop, input_x_gradient, deconv

def attribution_maps(model, inputs, labels, include_occlusion):
    num_labels = labels.shape[1]
    
    if include_occlusion == 1:
        saliency, guided_backprop, input_x_gradient, deconv, occlusion = attributions(model, include_occlusion)
        
        # Dictionary to store attribution maps for each method and each label
        attribution_maps_dict = {
            "Saliency": [],
            "Guided Backprop": [],
            "Input X Gradient": [],
            "Deconvolution": [],
            "Occlusion": []
        }
    
        # Compute attribution maps for each label
        for label_index in range(num_labels):
            target = torch.tensor([label_index]).to(inputs.device)
            saliency_maps = saliency.attribute(inputs, target=target)
            guided_backprop_maps = guided_backprop.attribute(inputs, target=target)
            input_x_gradient_maps = input_x_gradient.attribute(inputs, target=target)
            deconv_maps = deconv.attribute(inputs, target=target)
            occlusion_maps = occlusion.attribute(inputs, target=target, sliding_window_shapes=(3, 10, 10), 
                                     baselines=torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(DEVICE),
                                     strides=(3, 5, 5))
    
            attribution_maps_dict["Saliency"].append(saliency_maps)
            attribution_maps_dict["Guided Backprop"].append(guided_backprop_maps)
            attribution_maps_dict["Input X Gradient"].append(input_x_gradient_maps)
            attribution_maps_dict["Deconvolution"].append(deconv_maps)
            attribution_maps_dict["Occlusion"].append(occlusion_maps)
    
    else:
        saliency, guided_backprop, input_x_gradient, deconv = attributions(model, include_occlusion)
        
        # Dictionary to store attribution maps for each method and each label
        attribution_maps_dict = {
            "Saliency": [],
            "Guided Backprop": [],
            "Input X Gradient": [],
            "Deconvolution": []
        }
    
        # Compute attribution maps for each label
        for label_index in range(num_labels):
            target = torch.tensor([label_index]).to(inputs.device)
            saliency_maps = saliency.attribute(inputs, target=target)
            guided_backprop_maps = guided_backprop.attribute(inputs, target=target)
            input_x_gradient_maps = input_x_gradient.attribute(inputs, target=target)
            deconv_maps = deconv.attribute(inputs, target=target)
    
            attribution_maps_dict["Saliency"].append(saliency_maps)
            attribution_maps_dict["Guided Backprop"].append(guided_backprop_maps)
            attribution_maps_dict["Input X Gradient"].append(input_x_gradient_maps)
            attribution_maps_dict["Deconvolution"].append(deconv_maps)

    return attribution_maps_dict
