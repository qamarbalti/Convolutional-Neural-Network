# Convolutional-Neural-Network
1) Data Preparation :
1. Data Gathering:
Method Used: os.listdir() and iteration through subfolders.
Rationale: Iterated through the subfolders of the provided dataset path to gather file paths and labels for 
each image in the dataset.
2. Label Extraction:
Method Used: List comprehension and set conversion.
Rationale: Extracted unique labels from the gathered dataset to understand the diversity of classes 
present in the dataset.
3. Dataset Splitting:
Method Used: train_test_split() from sklearn.model_selection.
Rationale: Split the dataset into training, validation, and test sets to provide distinct subsets for model 
training, validation, and evaluation.
4. Subsetting Training Data:
Method Used: Additional train_test_split() for training set.
Rationale: Further split the training set into training and validation sets. This ensures that the model is 
trained on a larger portion of the data, with a smaller subset used for validation during training.
5. Directory Creation and File Movement:
Methods Used: os.makedirs() for directory creation and shutil.move() for file movement.
Rationale: Organized the dataset into separate directories for training, validation, and test sets. This 
structured approach facilitates efficient data loading during model training and evaluation.
6. Visualization:
Methods Used: matplotlib.pyplot and PIL.Image for image display.
Rationale: Displayed sample images from each subfolder within the training set to visually confirm the 
correctness of the data organization. This step aids in identifying any potential issues with the dataset 
structure.
7. Resulting Dataset Structure:
Directory Structure:
PTSD_Recognition
└── train_split
 ├── train
 │ ├── label_1
 │ │ └── image_1.jpg
 │ │ └── ...
 │ ├── label_2
 │ └── ...
 ├── val
 └── test
2) CNN Model Setup and Training Initialization
Convolutional Layers:
Reasoning:
Conv1: Increased kernel size and padding to capture more complex features in the input image. A larger 
receptive field helps in recognizing patterns at a broader scale.
Conv2 and Conv3: Followed by smaller kernel sizes, enabling the network to capture finer details in the 
deeper layers. Multiple convolutional layers allow the extraction of hierarchical features.
2. Pooling Layers:
Reasoning:
Pool1 and Pool3: Increased stride for downsampling the spatial dimensions more aggressively. This helps 
in reducing computational load and retaining essential features.
Pool2: Default stride to maintain a balance between downsampling and preserving spatial information.
3. Fully Connected Layers:
Reasoning:
Fc1: Adjusted the input size based on the flattened output from the convolutional layers. This layer acts 
as a feature aggregator, transforming the high-dimensional representation into a form suitable for 
classification.
Fc2: Final fully connected layer for classification into the specified number of classes.
4. Batch Normalization:
Reasoning:
Batch_norm: Applied after the second convolutional layer to normalize activations, improving 
convergence and generalization. It helps mitigate issues like internal covariate shift.
5. Dropout (Optional):
Reasoning:
Dropout: Introduces regularization by randomly dropping out a fraction of units during training. This 
helps prevent overfitting and enhances the model's ability to generalize to unseen data.
6. Weight Initialization:
Reasoning:
Initialize_weights(): Employed He initialization for convolutional and linear layers, promoting stable and 
efficient training. This initialization method is well-suited for ReLU activation functions.
7. Loss Function and Optimizer:
Reasoning:
Cross EntropyLoss: 
Suitable for multi-class classification tasks, as it combines the softmax activation with the negative loglikelihood loss.
Adam Optimizer: Adaptive optimization algorithm with momentum, which tends to converge faster and 
be less sensitive to learning rate tuning.
8. Model Overview:
The modified Simple CNN architecture is designed to handle complex features through varying kernel 
sizes, increased pooling strides, and adjusted fully connected layer input sizes.
Batch normalization aids in stabilizing training, dropout provides optional regularization, and weight 
initialization contributes to efficient learning.
Training and Optimization
1. Training Process:
Training Dataset:
Utilized the CIFAR-10 dataset for training, consisting of 60,000 32x32 color images in 10 different classes.
Transformed the input images using normalization and data augmentation techniques.
Model Architecture:
The modified Simple CNN consists of three convolutional layers with increased kernel sizes, padding, and 
pooling strides.
Fully connected layers were adjusted for appropriate input sizes.
Batch normalization and dropout were included for regularization.
2. Optimization Setup:
Loss Function:
Chose CrossEntropyLoss as the loss function, suitable for multi-class classification tasks.
Optimizer:
Used the Adam optimizer with a learning rate of 0.001 for adaptive learning and efficient convergence.
Learning Rate Scheduler:
Implemented a step-wise learning rate scheduler, reducing the learning rate by a factor of 0.1 every 7 
epochs.
3. Training Loop:
Epochs:
Trained the model for 10 epochs, iterating over the entire training dataset in each epoch.
Training and Validation Metrics:
Tracked training loss and accuracy for each epoch to monitor model performance.
Validated the model on a separate validation set to assess generalization.
Early Stopping (Not Implemented):
No early stopping mechanism was applied, but this can be added based on validation performance to 
prevent overfitting.
4. Training Results:
Training Accuracy:
Achieved increasing training accuracy, reaching 89.61% by the end of the training process.
Validation Accuracy:
Observed improvement in validation accuracy, reaching 77.09% after 10 epochs.
5. Loss Analysis:
Training Loss:
Decreased consistently, indicating effective learning and adaptation to the training dataset.
Validation Loss:
Remained relatively low, signifying good generalization performance on unseen data.
. Transfer Learning Setup:
Pre-trained Model:
Utilized a pre-trained ResNet18 model from torchvision.
Pre-trained models have learned features from large datasets (e.g., ImageNet), which can be beneficial 
for transfer learning.
Data Preparation:
Transformed the input images to the size expected by ResNet (224x224) and applied normalization.
Used the CIFAR-10 dataset for training and validation.
Model Modification:
Replaced the fully connected layer (fc) of the ResNet18 model to match the number of classes in the 
CIFAR-10 dataset (10 classes).
2. Optimization Setup:
Loss Function:
Employed CrossEntropyLoss, suitable for multi-class classification tasks.
Optimizer:
Utilized Adam optimizer with a learning rate of 0.001 for efficient convergence.
Learning Rate Scheduler:
Implemented a step-wise learning rate scheduler, reducing the learning rate by a factor of 0.1 every 7 
epochs.
GPU Support:
Checked for GPU availability and moved the model and criterion to the GPU for accelerated training.
3. Training Loop:
Epochs:
Trained the model for 2 epochs, leveraging the pre-trained features for faster convergence.
Training and Validation Metrics:
Monitored training loss, training accuracy, validation loss, and validation accuracy.
4. Training Results:
Training Accuracy:
Achieved training accuracy of approximately 99.25% after 2 epochs, showcasing effective learning and 
adaptation to the dataset.
Validation Accuracy:
Validation accuracy reached approximately 77.36%, indicating reasonable generalization performance.
Loss Analysis:
Training loss decreased significantly, indicating effective learning.
Validation loss remained reasonable, demonstrating good generalization.
5. Further Steps:
Further fine-tuning or increasing the number of training epochs may lead to improved performance.
Hyperparameter tuning and experimentation with different architectures can be explored.
Transfer Learning with Layer Freezing
Transfer Learning Setup:
Pre-trained Model:
Employed ResNet18 as the base pre-trained model from torchvision.
Utilized the CIFAR-10 dataset for training and validation.
Data Subset:
Created a subset of the CIFAR-10 dataset with a specified size for faster experimentation.
Adjusted the subset_size parameter based on the available computational resources.
2. Model Modification:
Freezing Layers:
Froze the first two layers of the ResNet18 model to retain pre-trained features.
Ensured that the weights of these layers remain fixed during training by setting requires_grad to False.
Fully Connected Layer Adjustment:
Modified the fully connected layer (fc) to align with the number of classes in the CIFAR-10 dataset (10 
classes).
3. Optimization Setup:
Loss Function:
Employed CrossEntropyLoss, suitable for multi-class classification tasks.
Optimizer:
Utilized Adam optimizer with a learning rate of 0.001 for efficient convergence.
Learning Rate Scheduler:
Implemented a step-wise learning rate scheduler, reducing the learning rate by a factor of 0.1 every 7 
epochs.
GPU Support:
Checked for GPU availability and moved the model and criterion to the GPU for accelerated training.
4. Training Loop:
Epochs:
Trained the model for 2 epochs, considering the smaller subset for faster experimentation.
Training and Validation Metrics:
Monitored training loss, training accuracy, validation loss, and validation accuracy.
5. Training Results:
Freezing Impact:
Freezing the first two layers preserved essential pre-trained features, enhancing model stability.
Ensured that these layers did not undergo significant changes during the limited training.
Training Accuracy:
Achieved high training accuracy, indicating effective learning of the model's unfrozen parameters.
Validation Accuracy:
Validation accuracy provides insights into the model's generalization performance with frozen layers.
Loss Analysis:
Monitored the training and validation loss to ensure that the model generalizes well on the smaller 
subset.
EVUALUATION AND OPTIMIZATION
1. Test Set Evaluation:
Test Accuracy:
The model achieved a test accuracy of 10.74%.
2. Confusion Matrix Analysis:
Confusion Matrix:
The confusion matrix provides a detailed breakdown of the model's predictions across different classes.
Each row represents the true class, and each column represents the predicted class.
True\Predicted Class 0 Class 1 Class 2 Class 3 Class 4 Class 5 Class 6 Class 7 Class 8 Class 9
Class 0 164 4 130 432 0 5 0 3 42 220
Class 1 420 2 3 354 0 2 0 5 16 198
Class 2 272 2 94 521 0 0 0 0 22 89
Class 3 297 1 75 506 0 2 1 8 22 88
Class 4 163 2 109 511 0 2 0 3 18 192
Class 5 360 0 27 508 0 1 0 3 13 88
Class 6 156 1 160 617 0 0 2 0 16 48
Class 7 211 3 53 475 0 3 0 3 16 236
Class 8 229 0 9 544 0 1 0 1 9 207
Class 9 216 0 1 470 0 0 0 0 20 293
3. Sample Predictions Analysis:
Observations:
Provided sample predictions for a limited number of samples from the test set.
Highlighted instances where the model's predictions align or differ from the true classes.
Sample Details:
Sample 1: True Class 3, Predicted Class 3
Sample 2: True Class 6, Predicted Class 3
Sample 3: True Class 5, Predicted Class 0
Sample 4: True Class 0, Predicted Class 3
Sample 5: True Class 3, Predicted Class (Incomplete prediction)
4. Analysis and Recommendations:
Low Test Accuracy:
The achieved test accuracy is relatively low, indicating potential challenges in generalization.
Confusion Matrix Insights:
Analyzed confusion matrix patterns to identify classes with frequent misclassifications.
Focused on areas requiring further investigation and potential model improvement.
Sample Predictions Examination:
Examined individual sample predictions to understand specific cases where the model struggle
