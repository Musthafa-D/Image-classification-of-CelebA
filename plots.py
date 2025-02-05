import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from captum.attr import Saliency, GuidedBackprop, InputXGradient, Deconvolution, Occlusion
from ccbdl.utils.logging import get_logger
from ccbdl.evaluation.plotting.base import GenericPlot
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from networks import NLRL_AO
from ccbdl.utils import DEVICE
from setsize import set_size
import matplotlib.ticker as ticker
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


class Accuracy_plot(GenericPlot):
    def __init__(self, learner):
        super(Accuracy_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating accuracy plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        ytr = self.learner.data_storage.get_item("train_acc")
        yatr = self.learner.data_storage.get_item("a_train_acc")
        yt = self.learner.data_storage.get_item("test_acc")

        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        ax.plot(x, ytr, label='$\\mathrm{Acc}_{\\mathrm{train}}$')
        ax.plot(x, yatr, label='$\\mathrm{Acc}_{\\mathrm{train\\_avg}}$')
        ax.plot(x, yt, label='$\\mathrm{Acc}_{\\mathrm{test}}$')

        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{Acc}$", fontsize=14)

        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=100)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(True)
        ax.set_yticks(range(10, 101, 10))
        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "accuracies"))
        return figs, names


class ConfusionMatrixPlot(GenericPlot):
    def __init__(self, learner):
        super(ConfusionMatrixPlot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating confusion matrix plot")
    
    def consistency_check(self):
        return True

    def plot(self):
        figs = []
        names = []
        
        label_names = ["Attractive", "Black_hair", "Heavy_Makeup", "High_Cheekbones", "Male", 
                       "No_Beard", "Smiling", "Wearing_Earrings", "Wearing_Lipstick", "Young"]
        
        for types in ["train", "test"]:
            predictions_list = self.learner.data_storage.get_item(f"{types}_predictions")
            true_labels_list = self.learner.data_storage.get_item(f"{types}_labels")
            epochs = self.learner.data_storage.get_item("epochs")
            
            for epoch_idx, epoch in enumerate(epochs):
                for i, label_name in enumerate(label_names):
                    true_labels = true_labels_list[epoch_idx]
                    predictions = predictions_list[epoch_idx]

                    # Generate the confusion matrix for the label
                    cm = confusion_matrix(true_labels[:, i], predictions[:, i])

                    # Plot the confusion matrix
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax,
                                xticklabels=['No ' + label_name, label_name],
                                yticklabels=['No ' + label_name, label_name])

                    # Save the plot
                    plot_filename = f"epoch_{epoch}"
                    plot_path = os.path.join("plots", "confusion_matrices", types, label_name, plot_filename)
                    
                    names.append(plot_path)
                    figs.append(fig)
                    
                    plt.close(fig)    
        return figs, names


class Precision_plot(GenericPlot):
    def __init__(self, learner):
        super(Precision_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating precision plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        ytr = self.learner.data_storage.get_item("train_prec")
        yt = self.learner.data_storage.get_item("test_prec")

        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
        
        ax.plot(x, ytr, label='$\\mathrm{Prec}_{\\mathrm{train}}$')
        ax.plot(x, yt, label='$\\mathrm{Prec}_{\\mathrm{test}}$')

        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{Prec}$", fontsize=14)

        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=1.05)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(True)
        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "precisions"))
        return figs, names


class Recall_plot(GenericPlot):
    def __init__(self, learner):
        super(Recall_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating recall plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        ytr = self.learner.data_storage.get_item("train_rec")
        yt = self.learner.data_storage.get_item("test_rec")

        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        ax.plot(x, ytr, label='$\\mathrm{Rec}_{\\mathrm{train}}$')
        ax.plot(x, yt, label='$\\mathrm{Rec}_{\\mathrm{test}}$')

        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{Rec}$", fontsize=14)
        
        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=1.05)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(True)
        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "recalls"))
        return figs, names


class F1_plot(GenericPlot):
    def __init__(self, learner):
        super(F1_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating f1 plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        ytr = self.learner.data_storage.get_item("train_f1s")
        yt = self.learner.data_storage.get_item("test_f1s")

        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        ax.plot(x, ytr, label='$\\mathrm{F1}_{\\mathrm{train}}$')
        ax.plot(x, yt, label='$\\mathrm{F1}_{\\mathrm{test}}$')

        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{F1}$", fontsize=14)
        
        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=1.05)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(True)
        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "f1s"))
        return figs, names


class Loss_plot(GenericPlot):
    def __init__(self, learner):
        super(Loss_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating loss plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        ytr = self.learner.data_storage.get_item("train_loss")
        yatr = self.learner.data_storage.get_item("a_train_loss")
        yt = self.learner.data_storage.get_item("test_loss")

        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        ax.plot(x, ytr, label='$\\mathcal{L}_{\\mathrm{train}}$')
        ax.plot(x, yatr, label='$\\mathcal{L}_{\\mathrm{train\\_avg}}$')
        ax.plot(x, yt, label='$\\mathcal{L}_{\\mathrm{test}}$')

        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel('$\\mathcal{L}$', fontsize=14)

        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=max(ytr))

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(True)
        ax.set_yticks(range(1, 3, 1))
        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "losses"))
        return figs, names


class TimePlot(GenericPlot):
    def __init__(self, learner):
        super(TimePlot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating time plot")

    def consistency_check(self):
        return True

    def plot(self):
        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
        
        xs, ys = zip(*self.learner.data_storage.get_item("Time", batch=True))
        
        ax.plot(xs, [y - ys[0]for y in ys])
        ax.set_xlabel('$B$', fontsize=14)
        ax.set_ylabel('$t$', fontsize=14)
        
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=max(ys))

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "time_plot"))    
        return figs, names


class Learning_rate_plot(GenericPlot):
    def __init__(self, learner):
        super(Learning_rate_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating learning rate plot")

    def consistency_check(self):
        return True

    def plot(self):
        figs = []
        names = []
        
        learning_rate = self.learner.data_storage.get_item("learning_rate")
        learning_rate_l = self.learner.data_storage.get_item("learning_rate_l")
        x = self.learner.data_storage.get_item("batch")

        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
        ax.plot(learning_rate, label="$\\mathrm{lr}_{\\mathrm{m}}$")
        ax.plot(learning_rate_l, label="$\\mathrm{lr}_{\\mathrm{f}}$")

        ax.set_xlabel('$B$', fontsize=14)
        ax.set_ylabel('$lr$', fontsize=14)

        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0)

        # Use ScalarFormatter to display y-axis in scientific notation (powers of 10)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "learning_rate_schedule"))
        return figs, names


class Probabilities_plot(GenericPlot):
    def __init__(self, learner):
        super(Probabilities_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating histogram of probabilities plots")

    def consistency_check(self):
        return True
    
    def values(self, types):
        inputs_list = self.learner.data_storage.get_item(f"{types}_inputs")
        output_list = self.learner.data_storage.get_item(f"{types}_outputs")
        labels_list = self.learner.data_storage.get_item(f"{types}_actual_label")
        predicted_list = self.learner.data_storage.get_item(f"{types}_predicted_label")

        epoch = self.learner.learner_config['num_epochs'] - 1
        
        inputs = inputs_list[-1][:16]
        outputs = output_list[-1][:16]
        labels = labels_list[-1][:16]
        preds = predicted_list[-1][:16]
        
        return inputs, outputs, labels, preds, epoch
    
    def plot(self):
        names = []
        figs = []
    
        label_names = ["Attractive", "Black_hair", "Heavy_Makeup", "High_Cheekbones", "Male", 
                       "No_Beard", "Smiling", "Wearing_Earrings", "Wearing_Lipstick", "Young"]
        max_images_per_plot = 4  # Maximum number of images per plot
    
        for types in ["train", "test"]:
            inputs, outputs, labels, preds, epoch = self.values(types)
    
            sigmoid = torch.nn.Sigmoid()
            sigmoid_outputs = sigmoid(outputs)
    
            # Process the data in chunks of max_images_per_plot
            num_chunks = len(inputs) // max_images_per_plot + (len(inputs) % max_images_per_plot > 0)
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * max_images_per_plot
                end_idx = start_idx + max_images_per_plot
                chunk_inputs = inputs[start_idx:end_idx]
                chunk_outputs = sigmoid_outputs[start_idx:end_idx]
                chunk_labels = labels[start_idx:end_idx]
                chunk_preds = preds[start_idx:end_idx]
    
                # Creating subplots for each chunk
                fig, axs = plt.subplots(2, len(chunk_inputs), figsize=(10 * len(chunk_inputs), 10), squeeze=False)
                fig.suptitle(f'Histogram of probabilities of labels\n{label_names}')
    
                for i in range(len(chunk_inputs)):
                    img = chunk_inputs[i].cpu().detach().permute(1, 2, 0).numpy()
                    label = chunk_labels[i].cpu().detach().numpy()
                    pred = chunk_preds[i].cpu().detach().numpy()
                    output_prob = chunk_outputs[i].cpu().detach().numpy()
    
                    axs[0, i].imshow(img)
                    axs[0, i].set_title(f"Image {start_idx + i + 1}")
                    axs[0, i].axis("off")
    
                    axs[1, i].bar(range(len(output_prob)), output_prob, color=['green' if pred[j] == label[j] else 'blue' for j in range(len(label))])
                    title_str = f"Label: {label}\nPred: {pred}"
                    axs[1, i].set_title(title_str, fontsize=12)
                    axs[1, i].set_xticks(range(len(output_prob)))
                    axs[1, i].set_xticklabels(label_names, rotation=45, ha="right")
                    axs[1, i].set_ylim((0, 1))
                    axs[1, i].set_yticks(np.arange(0, 1.1, 0.1))
                
                # Adjust legend for cases with fewer subplots
                correct_bar = plt.Rectangle((0,0),1,1,fc='green', edgecolor='none')
                incorrect_bar = plt.Rectangle((0,0),1,1,fc='blue', edgecolor='none')
                fig.legend([correct_bar, incorrect_bar], ['Correct', 'Incorrect'], loc='upper right', ncol=2)
    
                names.append(os.path.join("plots", "analysis_plots", "probabilities_plots", f"{types}_data", f"subset_{chunk_idx + 1}"))
                figs.append(fig)
                plt.close(fig)
    
        return figs, names


class Attribution_plots(GenericPlot):
    def __init__(self, learner):
        super(Attribution_plots, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribution maps")

    def consistency_check(self):
        return True
    
    def safe_visualize(self, attr, original_image, title, fig, ax, types, cmap):
        if not (attr == 0).all():
            viz.visualize_image_attr(attr, 
                                     original_image=original_image, 
                                     method='heat_map', 
                                     sign='all', 
                                     show_colorbar=True, 
                                     title=title, 
                                     plt_fig_axis=(fig, ax),
                                     cmap=cmap)
        else:
            print(f"Skipping visualization for {types} data for the attribution: {title} as all attribute values are zero.")

    def plot(self):
        names = []
        figs = []
        
        cmap = LinearSegmentedColormap.from_list("BlWhGn", ["blue", "white", "green"])
        
        max_images_per_plot = 4
        imp_values = Probabilities_plot(self.learner)
        
        self.learner._load()
        model = self.learner.model

        label_names = ["Attractive", "Black_hair", "Heavy_Makeup", "High_Cheekbones", "Male", 
                       "No_Beard", "Smiling", "Wearing_Earrings", "Wearing_Lipstick", "Young"]
        
        for types in ["train", "test"]:
            inputs, outputs, labels, preds, epoch = imp_values.values(types)
            attr_maps_dict = attribution_maps(model, inputs, preds)
            
            num_attribution_methods = len(attr_maps_dict.keys())

            # Process the data in chunks of max_images_per_plot
            num_chunks = len(inputs) // max_images_per_plot + (len(inputs) % max_images_per_plot > 0)
            for label_idx, label_name in enumerate(label_names):
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * max_images_per_plot
                    end_idx = start_idx + max_images_per_plot
                    chunk_inputs = inputs[start_idx:end_idx]
                    chunk_labels = labels[start_idx:end_idx]
                    chunk_preds = preds[start_idx:end_idx]
                    
                    # Define the figure and subplots for each label
                    fig, axs = plt.subplots(max_images_per_plot, num_attribution_methods + 1, figsize=(20, 10 * max_images_per_plot))

                    for idx, img_input in enumerate(chunk_inputs):
                        img = img_input.cpu().detach().permute(1, 2, 0).numpy()
                        label = chunk_labels[idx].cpu().detach().numpy()
                        pred = chunk_preds[idx].cpu().detach().numpy()
                        
                        # Convert binary label and prediction to named labels
                        actual_labels = [name for i, name in enumerate(label_names) if label[i] == 1]
                        predicted_labels = [name for i, name in enumerate(label_names) if pred[i] == 1]
                        actual_labels_str = ", \n".join(actual_labels)
                        predicted_labels_str = ", \n".join(predicted_labels)
                        
                        axs[idx, 0].imshow(img)
                        axs[idx, 0].set_title(f"Image {start_idx + idx + 1}\nActual: {actual_labels_str}\nPredicted: {predicted_labels_str}")
                        axs[idx, 0].axis("off")

                        for method_idx, (method_name, attr_maps_list) in enumerate(attr_maps_dict.items()):
                            attr_map = attr_maps_list[label_idx]
                            attr_map = attr_maps_list[label_idx]
                            
                            result = np.transpose(attr_map[start_idx + idx].squeeze().cpu().detach().numpy(), (1, 2, 0))
                            title_str = f"{method_name}"
                            ax = axs[idx, method_idx + 1]
                            self.safe_visualize(result, img, title_str, fig, ax, types, cmap)
                            ax.axis("off")

                    fig.suptitle(f"Attribution plots for '{label_name}' - {types.title()} Data")
                    
                    names.append(os.path.join("plots", "analysis_plots", "attribution_plots", f"{types}_data", f"{label_name}_subset_{chunk_idx + 1}"))
                    figs.append(fig)

                    plt.close(fig)
        return figs, names
    

def attributions(model):
    # Initialize the Saliency object
    saliency = Saliency(model)
    # Initialize the GuidedBackprop object
    guided_backprop = GuidedBackprop(model)
    # Initialize the DeepLift object
    input_x_gradient = InputXGradient(model)
    # Initialize the Deconvolution object
    deconv = Deconvolution(model)
    # Initialize the Occlusion object
    occlusion = Occlusion(model)
    
    return saliency, guided_backprop, input_x_gradient, deconv, occlusion

def attribution_maps(model, inputs, labels):
    num_labels = labels.shape[1]
    saliency, guided_backprop, input_x_gradient, deconv, occlusion = attributions(model)
    
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
        # Extracting the specific label for all inputs
        # print(f"{label_index}")
        # print(f"labels: {labels}\n")
        target = labels[:, label_index].long()
        target = target.to(inputs.device)
        # print(f"target: {target}\n")
        # print(f"inputs: {inputs}\n")
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

    return attribution_maps_dict


class Hist_plot(GenericPlot):
    def __init__(self, learner):
        super(Hist_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating histogram of nlrl_ao plot")

    def consistency_check(self):
        return True
    
    def extract_parameters(self, model):
        for layer in model.modules():
            if isinstance(layer, NLRL_AO):
                negation = torch.sigmoid(layer.negation).detach().cpu().numpy()
                relevancy = torch.sigmoid(layer.relevancy).detach().cpu().numpy()
                selection = torch.sigmoid(layer.selection).detach().cpu().numpy()
    
                negation_init = torch.sigmoid(layer.negation_init).detach().cpu().numpy()
                relevancy_init = torch.sigmoid(layer.relevancy_init).detach().cpu().numpy()
                selection_init = torch.sigmoid(layer.selection_init).detach().cpu().numpy()
                
                return (negation, relevancy, selection), (negation_init, relevancy_init, selection_init)
        return None

    def plot(self):
        figs=[]
        names=[]
        
        bool_ops = ['negation', 'relevancy', 'selection']
        # Load the classifier  
        for models in ["initial", "best"]:
            if models == "initial":
                model = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
            else:
                model = self.learner._load_best()  # Load the best epoch's model with the respective weights
            params, init_params = self.extract_parameters(model)
        
            for i, (param, init_param) in enumerate(zip(params, init_params)):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(init_param.ravel(), color='blue', alpha=0.5, bins=np.linspace(0, 1, 30), label='Initial')
                ax.hist(param.ravel(), color='red', alpha=0.5, bins=np.linspace(0, 1, 30), label='Trained')
                
                ax.set_xlabel('$\sigma(W)$', fontsize=14) # sigmoid of the learnable weight matrices
                ax.set_ylabel('$|W|$', fontsize=14) # number of parameters
                ax.legend(loc='upper right')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                
                figs.append(fig)
                plt.close(fig)
                names.append(os.path.join("plots", "histogram_plots", f"{models}_{bool_ops[i]}"))
        return figs, names
    

"""updated plots"""
class Probabilities_plot_update(GenericPlot):
    def __init__(self, learner):
        super(Probabilities_plot_update, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating histogram of probabilities plots")

    def consistency_check(self):
        return True
    
    def values(self, types):
        inputs_list = self.learner.data_storage.get_item(f"{types}_inputs")
        labels_list = self.learner.data_storage.get_item(f"{types}_actual_label")
        
        inputs = inputs_list[-1][:8]
        labels = labels_list[-1][:8]
        
        return inputs, labels

    
    def plot(self):
        names = []
        figs = []
    
        label_names = ["Attractive", "Black_hair", "Heavy_Makeup", "High_Cheekbones", "Male", 
                       "No_Beard", "Smiling", "Wearing_Earrings", "Wearing_Lipstick", "Young"]
        
        max_images_per_plot = 2  # Maximum number of images per plot
    
        for types in ["train", "test"]:
            inputs, labels = self.values(types)
        
            for models in ["initial", "best"]:   
                    if models == "initial":
                        model_nlrl, model_linear = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
                    else:
                        model_nlrl, model_linear = self.learner._load_best()  # Load the best epoch's model with the respective weights
                    
                    for layers in ["nlrl", "linear"]:
                        if layers == "nlrl":
                            model = model_nlrl
                        else:
                            model = model_linear
        
                        outputs = model(inputs)
                        sigmoid = torch.nn.Sigmoid()
                        sigmoid_outputs = sigmoid(outputs)
                        preds = (outputs.detach() > 0.5).float()
                
                        # Process the data in chunks of max_images_per_plot
                        num_chunks = len(inputs) // max_images_per_plot + (len(inputs) % max_images_per_plot > 0)
                        for chunk_idx in range(num_chunks):
                            start_idx = chunk_idx * max_images_per_plot
                            end_idx = start_idx + max_images_per_plot
                            chunk_inputs = inputs[start_idx:end_idx]
                            chunk_outputs = sigmoid_outputs[start_idx:end_idx]
                            chunk_labels = labels[start_idx:end_idx]
                            chunk_preds = preds[start_idx:end_idx]
                
                            # Creating subplots for each chunk
                            fig, axs = plt.subplots(2, len(chunk_inputs), figsize=(10 * len(chunk_inputs), 10), squeeze=False)
                            # fig.suptitle(f'Histogram of probabilities of labels\n{label_names}')
                
                            for i in range(len(chunk_inputs)):
                                img = chunk_inputs[i].cpu().detach().permute(1, 2, 0).numpy()
                                label = chunk_labels[i].cpu().detach().numpy()
                                pred = chunk_preds[i].cpu().detach().numpy()
                                output_prob = chunk_outputs[i].cpu().detach().numpy()
                
                                axs[0, i].imshow(img)
                                axs[0, i].axis("off")
                
                                axs[1, i].bar(range(len(output_prob)), output_prob, color=['green' if pred[j] == label[j] else 'blue' for j in range(len(label))])
                                title_str = f"A: {label}\nP: {pred}"
                                axs[1, i].set_title(title_str, fontsize=17)
                                axs[1, i].set_xticks(range(len(output_prob)))
                                axs[1, i].set_xticklabels(label_names, rotation=45, ha="right")
                                axs[1, i].set_ylim((0, 1))
                                axs[1, i].set_yticks(np.arange(0, 1.1, 0.1))
                            
                            # Adjust legend for cases with fewer subplots
                            # correct_bar = plt.Rectangle((0,0),1,1,fc='green', edgecolor='none')
                            # incorrect_bar = plt.Rectangle((0,0),1,1,fc='blue', edgecolor='none')
                            # fig.legend([correct_bar, incorrect_bar], ['Correct', 'Incorrect'], loc='upper right', ncol=2)
                
                            names.append(os.path.join("plots", "analysis_plots", "probabilities_plots", f"{types}_data", f"{layers}", f"{models}_subset_{chunk_idx + 1}"))
                            figs.append(fig)
                            plt.close(fig)
    
        return figs, names


class Attribution_plots_update(GenericPlot):
    def __init__(self, learner):
        super(Attribution_plots_update, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribution maps")

    def consistency_check(self):
        return True
    
    def safe_visualize(self, attr, original_image, title, fig, ax, types, cmap, check):
        if not (attr == 0).all():
            if len(attr.shape) == 2:
                attr = np.expand_dims(attr, axis=2)
            if check == 0:
                viz.visualize_image_attr(attr,
                                         method='heat_map',
                                         sign='all',
                                         title=title,
                                         plt_fig_axis=(fig, ax),
                                         use_pyplot=False,
                                         fontsize=17,
                                         cmap=cmap)
            else:
                viz.visualize_image_attr(attr,
                                         method='heat_map',
                                         sign='all',
                                         plt_fig_axis=(fig, ax),
                                         use_pyplot=False,
                                         fontsize=17,
                                         cmap=cmap)
        else:
            print(f"Skipping visualization for {types} data for the attribution: {title} as all attribute values are zero.")

    def plot(self):
        names = []
        figs = []
        
        cmap = LinearSegmentedColormap.from_list("BlWhGn", ["blue", "white", "green"])
        
        max_images_per_plot = 1  # Set to plot 1 image per plot
        imp_values = Probabilities_plot_update(self.learner)

        label_names = ["Attractive", "Black_hair", "Heavy_Makeup", "High_Cheekbones", "Male", 
                       "No_Beard", "Smiling", "Wearing_Earrings", "Wearing_Lipstick", "Young"]
        
        for types in ["train", "test"]:
            inputs, labels = imp_values.values(types)
            
            for models in ["initial", "best"]:   
                if models == "initial":
                    model_nlrl, model_linear = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
                else:
                    model_nlrl, model_linear = self.learner._load_best()  # Load the best epoch's model with the respective weights
                
                for layers in ["nlrl", "linear"]:
                    model = model_nlrl if layers == "nlrl" else model_linear
                    
                    outputs = model(inputs)
                    preds = (outputs.detach() > 0.5).float()
        
                    attr_maps_dict = attribution_maps(model, inputs, preds)
                    
                    num_attribution_methods = len(attr_maps_dict.keys())
        
                    # Process the data in chunks of max_images_per_plot (now set to 1)
                    num_chunks = len(inputs) // max_images_per_plot + (len(inputs) % max_images_per_plot > 0)
                    for label_idx, label_name in enumerate(label_names):
                        for chunk_idx in range(num_chunks):
                            start_idx = chunk_idx * max_images_per_plot
                            end_idx = start_idx + max_images_per_plot
                            chunk_inputs = inputs[start_idx:end_idx]
                            chunk_labels = labels[start_idx:end_idx]
                            chunk_preds = preds[start_idx:end_idx]
                            
                            # Define the figure and subplots for each label (only 1 row since max_images_per_plot is 1)
                            fig, axs = plt.subplots(1, num_attribution_methods + 1, figsize=(15, 8))  # Single row for one image

                            img_input = chunk_inputs[0]
                            img = img_input.cpu().detach().permute(1, 2, 0).numpy()
                            label = chunk_labels[0].cpu().detach().numpy()
                            pred = chunk_preds[0].cpu().detach().numpy()
                            
                            # Convert binary label and prediction to named labels
                            actual_labels = [name for i, name in enumerate(label_names) if label[i] == 1]
                            predicted_labels = [name for i, name in enumerate(label_names) if pred[i] == 1]
                            actual_labels_str = ", \n".join(actual_labels)
                            predicted_labels_str = ", \n".join(predicted_labels)
                            
                            axs[0].imshow(img)
                            axs[0].set_title(f"A: {actual_labels_str}\nP: {predicted_labels_str}")
                            axs[0].axis("off")

                            for method_idx, (method_name, attr_maps_list) in enumerate(attr_maps_dict.items()):
                                attr_map = attr_maps_list[label_idx]
                                
                                result = np.transpose(attr_map[start_idx].squeeze().cpu().detach().numpy(), (1, 2, 0))
                                title_str = f"{method_name}"
                                ax = axs[method_idx + 1]
                                
                                self.safe_visualize(result, img, title_str, fig, ax, types, cmap, check=0)
                                ax.axis("off")
                            
                            # Add a single colorbar for all subplots below the grid
                            fig.subplots_adjust(bottom=0.15)
                            cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
                            norm = plt.Normalize(vmin=-1, vmax=1)
                            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                            sm.set_array([])
                            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                            cbar.ax.tick_params(labelsize=17)  # Set colorbar tick label font size
                            
                            names.append(os.path.join("plots", "analysis_plots", "attribution_plots", f"{layers}", f"{types}_data", f"{label_name}_{models}_subset_{chunk_idx + 1}"))
                            figs.append(fig)

                            plt.close(fig)
        return figs, names


class Hist_plot_update(GenericPlot):
    def __init__(self, learner):
        super(Hist_plot_update, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating histogram of nlrl_ao plot")

    def consistency_check(self):
        return True
    
    def extract_parameters(self, model):
        for layer in model.modules():
            if isinstance(layer, NLRL_AO):
                negation = torch.sigmoid(layer.negation).detach().cpu().numpy()
                relevancy = torch.sigmoid(layer.relevancy).detach().cpu().numpy()
                selection = torch.sigmoid(layer.selection).detach().cpu().numpy()
    
                negation_init = torch.sigmoid(layer.negation_init).detach().cpu().numpy()
                relevancy_init = torch.sigmoid(layer.relevancy_init).detach().cpu().numpy()
                selection_init = torch.sigmoid(layer.selection_init).detach().cpu().numpy()
                
                return (negation, relevancy, selection), (negation_init, relevancy_init, selection_init)
        return None


    def plot(self):
        figs=[]
        names=[]
        
        bool_ops = ['negation', 'relevancy', 'selection']
        # Load the classifier  
        for models in ["initial", "best"]:
            if models == "initial":
                model, _ = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
            else:
                model, _ = self.learner._load_best()  # Load the best epoch's model with the respective weights
            params, init_params = self.extract_parameters(model)
        
            for i, (param, init_param) in enumerate(zip(params, init_params)):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(init_param.ravel(), color='blue', alpha=0.5, bins=np.linspace(0, 1, 30), label='Initial')
                ax.hist(param.ravel(), color='red', alpha=0.5, bins=np.linspace(0, 1, 30), label='Trained')
                
                ax.set_xlabel('$\sigma(W)$', fontsize=14) # sigmoid of the learnable weight matrices
                ax.set_ylabel('$|W|$', fontsize=14) # number of parameters
                ax.set_xlim(left=0, right=1)
                ax.set_ylim(bottom=0)
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                ax.legend(loc='upper right')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                
                figs.append(fig)
                plt.close(fig)
                names.append(os.path.join("plots", "nlrl", f"{models}_{bool_ops[i]}"))
        return figs, names


class Linear_plot_update(GenericPlot):
    def __init__(self, learner):
        super(Linear_plot_update, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating histogram and feature importance plots of linear layer")

    def consistency_check(self):
        return True
    
    def plot(self):
        figs = []
        names = []
        
        # Load the classifier  
        for models in ["initial", "best"]:
            if models == "initial":
                _, model = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
            else:
                _, model = self.learner._load_best()  # Load the best epoch's model with the respective weights
            
            # Access the final layer using -1 index
            linear_layer = model.model[-1]
            
            if isinstance(linear_layer, torch.nn.Linear):
                weights = linear_layer.weight.detach().cpu().numpy()
                biases = linear_layer.bias.detach().cpu().numpy().ravel()
                
                # Separate plot for weights histogram
                fig_weights, ax_weights = plt.subplots(figsize=(10, 6))
                ax_weights.hist(weights.ravel(), bins=np.linspace(np.min(weights), np.max(weights), 30), color='blue', alpha=0.7)
                # ax_weights.set_title('Weights Distribution')
                ax_weights.set_xlabel('$W$', fontsize=14) # Weight values
                ax_weights.set_ylabel('$f$', fontsize=14) # Frequency
                plt.tight_layout()
                
                figs.append(fig_weights)
                plt.close(fig_weights)
                names.append(os.path.join("plots", "linear", f"weights_{models}"))

                # Separate plot for biases histogram
                fig_biases, ax_biases = plt.subplots(figsize=(10, 6))
                ax_biases.hist(biases, bins=np.linspace(np.min(biases), np.max(biases), 30), color='green', alpha=0.7)
                # ax_biases.set_title('Biases Distribution')
                ax_biases.set_xlabel('$b$', fontsize=14) # Bias values
                ax_biases.set_ylabel('$f$', fontsize=14) # Frequency
                plt.tight_layout()
                
                figs.append(fig_biases)
                plt.close(fig_biases)
                names.append(os.path.join("plots", "linear", f"biases_{models}"))

                # Separate plot for in-features importance
                fig_in_features, ax_in_features = plt.subplots(figsize=(10, 6))
                
                # Calculate the importance (absolute magnitude of weights)
                in_feature_importance = np.abs(weights).sum(axis=0)
                
                ax_in_features.bar(range(len(in_feature_importance)), in_feature_importance)
                ax_in_features.set_xlabel('$x$', fontsize=14) # Input Features
                ax_in_features.set_ylabel('$\sum(|W|)$', fontsize=14) # Feature Importance (Sum of Absolute Weights)
                # ax_in_features.set_title('Feature Importance for Input Features (In-Features)')
                plt.tight_layout()
                
                figs.append(fig_in_features)
                plt.close(fig_in_features)
                names.append(os.path.join("plots", "linear", f"in_feature_importance_plot_{models}"))

                # Separate plot for out-features importance
                fig_out_features, ax_out_features = plt.subplots(figsize=(10, 6))
                
                # Calculate the importance (absolute magnitude of weights for each output feature)
                out_feature_importance = np.abs(weights).sum(axis=1)
                
                ax_out_features.bar(range(len(out_feature_importance)), out_feature_importance)
                ax_out_features.set_xlabel('$y$', fontsize=14)
                ax_out_features.set_ylabel('$\sum(|W|)$', fontsize=14)
                ax_out_features.set_xticks(np.arange(0, 10, 1))
                # ax_out_features.set_title('Feature Importance for Output Features (Out-Features)')
                plt.tight_layout()
                
                figs.append(fig_out_features)
                plt.close(fig_out_features)
                names.append(os.path.join("plots", "linear", f"out_feature_importance_plot_{models}"))
            
            else:
                self.logger.error(f"Layer at index -1 is not a linear layer. Found {type(linear_layer)} instead.")
        
        return figs, names


class Tsne_plot_update(GenericPlot):
    def __init__(self, learner):
        super(Tsne_plot_update, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots based on classifier's features and classifier's decision")
    
    def consistency_check(self):
        return True
    
    def get_features(self, classifier, imgs, layer):
        activation = {}
        
        def get_activation(name):
            def hook(classifier, inp, output):
                activation[name] = output.detach()
            return hook
        
        # Register the hook
        if layer == 'nlrl':
            handle = classifier.model[-2].register_forward_hook(get_activation('conv'))
        else:
            handle = classifier.model[-1].register_forward_hook(get_activation('conv'))
        _ = classifier(imgs)
        
        # Remove the hook
        handle.remove()
        return activation['conv']
    
    def compute_tsne(self, features):
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(features)
        return tsne_results
    
    def process_images(self, data_loader, classifier, cat, layer):
        all_features = []
        all_labels = []
        
        for imgs in data_loader:
            outputs = classifier(imgs)
            predicted_labels = (outputs.detach() > 0.5).float()
            if layer == "nlrl":
                features = self.get_features(classifier, imgs, "nlrl")
            else:
                features = self.get_features(classifier, imgs, "linear")
            features = features.view(features.size(0), -1)  # Flatten the features
            all_features.append(features)
            all_labels.append(predicted_labels)
            
        # Concatenate all the features and labels from the batches
        if cat == 1:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        return all_features, all_labels
    
    def plot(self):
        figs, names = [], []
        # Load the classifier  
        for models in ["initial", "best"]:
            if models == "initial":
                classifier_nlrl, classifier_linear = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
            else:
                classifier_nlrl, classifier_linear = self.learner._load_best()  # Load the best epoch's model with the respective weights
        
            # Setting concatenation true by initializing value as 1
            cat = 1
            
            epochs = self.learner.data_storage.get_item("epochs_gen")
            total = len(epochs)
            
            for types in ["train", "test"]:
                total_images = self.learner.data_storage.get_item(f"{types}_inputs")
                
                # Limit the number of test images for quicker plotting
                if types == "test":
                    total_images = total_images[:len(total_images) // 2]
                batches_per_epoch = int(len(total_images)/total)

                images = total_images[-batches_per_epoch:]
                images = torch.cat(images)
                dataset = ImageTensorDataset(images)
                data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
                
                attribute_names = ["Attractive", "Black_hair", "Heavy_Makeup", "High_Cheekbones", "Male", 
                                   "No_Beard", "Smiling", "Wearing_Earrings", "Wearing_Lipstick", "Young"]

                palette = ['red', 'blue']  # Two colors: one for 1 (attribute present), one for 0 (attribute absent)

                for layers in ["nlrl", "linear"]:
                    if layers == "nlrl":
                        features, labels = self.process_images(data_loader, classifier_nlrl, cat, "nlrl")
                    else:
                        features, labels = self.process_images(data_loader, classifier_linear, cat, "linear")
                    
                    tsne_results = self.compute_tsne(features.cpu().numpy())                

                    # Plot t-SNE for each attribute separately
                    for attr_idx, attr_name in enumerate(attribute_names):
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Separate points with attribute 1 vs attribute 0
                        attr_present = (labels[:, attr_idx] == 1).cpu().numpy()
                        attr_absent = (labels[:, attr_idx] == 0).cpu().numpy()

                        # Plot attribute present points
                        sns.scatterplot(
                            ax=ax,
                            x=tsne_results[attr_present, 0],
                            y=tsne_results[attr_present, 1],
                            color=palette[0],
                            # label=f"{attr_name}: Present",
                            alpha=0.5
                        )

                        # Plot attribute absent points
                        sns.scatterplot(
                            ax=ax,
                            x=tsne_results[attr_absent, 0],
                            y=tsne_results[attr_absent, 1],
                            color=palette[1],
                            # label=f"{attr_name}: Absent",
                            alpha=0.5
                        )

                        # ax.set_title(f"t-SNE plot for {attr_name} - {types.title()} Data ({layers} Layer)")
                        ax.legend()

                        figs.append(fig)
                        plt.close(fig)

                        # Save plot name
                        names.append(os.path.join("plots", "analysis_plots", "tsne_plots_update_celeba", f"{layers}",
                                                  f"{models}_{types}_{attr_name}_tsne_plot"))
                        
        return figs, names


# Custom Dataset class to handle lists of tensors
class ImageTensorDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]