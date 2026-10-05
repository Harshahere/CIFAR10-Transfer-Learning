# CIFAR10-Transfer-Learning
CIFAR-10 image classification using CNNs and MobileNetV2 transfer learning
# CIFAR-10 Image Classification using Transfer Learning

A deep learning project comparing a custom CNN trained from scratch with ImageNet-pretrained MobileNetV2 for CIFAR-10 image classification. The project explores data augmentation, transfer learning, feature extraction, selective fine-tuning, model evaluation, and error analysis.

## Overview

The objective of this project is to classify CIFAR-10 images into 10 object categories using convolutional neural networks and transfer learning.

Three different approaches were implemented and compared:

1. CNN trained from scratch
2. MobileNetV2 with frozen ImageNet-pretrained features
3. MobileNetV2 with selective fine-tuning

The final fine-tuned MobileNetV2 achieved **88.51% test accuracy**, compared with **70.81%** for the CNN baseline.

### Performance Improvement

CNN Baseline
70.81%

        ↓ +15.00 percentage points

MobileNetV2 Feature Extraction
85.91%

        ↓ +2.60 percentage points

MobileNetV2 Fine-Tuning
88.51%

Overall, transfer learning and fine-tuning improved test accuracy by:
+17.70 percentage points

Dataset
The project uses the CIFAR-10 dataset.
Dataset Characteristics
- 60,000 RGB images
- Image resolution: 32 × 32
- 50,000 training images
- 10,000 test images
- 10 classes
- 6,000 images per class
Classes
Class
Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck


Each image contains three color channels:
32 × 32 × 3

Project Objectives
- Build a CNN image-classification baseline from scratch.
- Understand convolutional neural network architectures.
- Apply data augmentation to improve generalization.
- Implement transfer learning using ImageNet-pretrained MobileNetV2.
- Understand feature extraction vs. fine-tuning.
- Selectively fine-tune deeper layers of a pretrained network.
- Evaluate classification performance using accuracy, precision, recall, and F1-score.
- Analyze model errors using a confusion matrix and misclassified images.
- Investigate overfitting during fine-tuning.
1. Data Preparation
The CIFAR-10 Python binary batches were loaded and converted into NumPy arrays.
The original CIFAR-10 representation stores each image as a flattened vector:
3072 = 32 × 32 × 3

The data was reshaped into:
(N, 32, 32, 3)

Normalization
Pixel values originally ranged from 0 to 255.
They were converted to float32 and normalized to the range [0, 1]:
X_train = X_train.astype("float32") / 255.0X_test = X_test.astype("float32") / 255.0


Train / Validation / Test Split
The original 50,000 training images were split into:
- Training: 45,000 images
- Validation: 5,000 images
- Test: 10,000 images
A stratified split with random_state=42 was used to preserve class proportions.
2. Data Augmentation
Data augmentation was used to improve generalization and reduce overfitting.
The baseline CNN used:
- Random horizontal flipping
- Random translations
The transformations were applied during training.
tf.keras.layers.RandomFlip("horizontal")tf.keras.layers.RandomTranslation(    height_factor=0.1,    width_factor=0.1)


3. CNN Baseline
A custom CNN was first trained from scratch to establish a baseline for comparison.
Architecture
Input: 32 × 32 × 3
        ↓
Random Horizontal Flip
        ↓
Random Translation
        ↓
Conv2D: 32 filters, 3×3, ReLU
        ↓
MaxPooling2D: 2×2
        ↓
Conv2D: 64 filters, 3×3, ReLU
        ↓
MaxPooling2D: 2×2
        ↓
Flatten
        ↓
Dense: 128, ReLU
        ↓
Dense: 10, Softmax

Model Parameters
The baseline CNN contains:
545,098 trainable parameters
Parameter Breakdown
Layer	Parameters
Conv2D (32)	896
Conv2D (64)	18,496
Dense (128)	524,416
Dense (10)	1,290
Total	545,098


Training Configuration
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Batch size: 64
- Epochs: 10
Training Result
Final training accuracy: 69.33%
Final validation accuracy: 70.42%
Test Result
Test Accuracy: 70.81%
Test Loss: 0.8460
This model served as the baseline against which transfer learning was evaluated.
4. Transfer Learning with MobileNetV2
The second approach used MobileNetV2 pretrained on ImageNet.
The original ImageNet classification head was removed using:
include_top=False


A new classification head was then added for the 10 CIFAR-10 classes.
Architecture
CIFAR-10 Image
32 × 32 × 3
        ↓
Resize
96 × 96 × 3
        ↓
Rescaling
[0,1] → [-1,1]
        ↓
MobileNetV2
ImageNet Pretrained
        ↓
GlobalAveragePooling2D
        ↓
Dense: 10, Softmax

MobileNetV2 Configuration
tf.keras.applications.MobileNetV2(    weights="imagenet",    include_top=False,    input_shape=(96, 96, 3))


The MobileNetV2 backbone contains:
2,257,984 parameters
During feature extraction:
- Trainable parameters: 12,810
- Non-trainable parameters: 2,257,984
The trainable parameters belong to the new CIFAR-10 classification head.
5. Input Preprocessing for MobileNetV2
The original CIFAR-10 images are 32 × 32, while the MobileNetV2 model used in this project receives 96 × 96 inputs.
Therefore, resizing was performed inside the model:
tf.keras.layers.Resizing(96, 96)


Since the CIFAR-10 images had already been normalized to [0, 1], MobileNetV2 preprocessing was implemented using:
tf.keras.layers.Rescaling(    scale=2.0,    offset=-1.0)


This maps:
0   → -1
0.5 →  0
1   → +1

6. Feature Extraction
In the first transfer-learning stage, the MobileNetV2 backbone was frozen.
MobileNetV2 → Frozen
Classification Head → Trainable

Only the newly added 10-class classifier was trained.
Global Average Pooling
MobileNetV2 produced feature maps of:
3 × 3 × 1280

GlobalAveragePooling2D converted these feature maps into:
1280

features.
Instead of flattening:
3 × 3 × 1280 = 11,520 values

Global Average Pooling produces:
1280 values

This significantly reduces the size of the classification head and helps control overfitting.
Training Configuration
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 64
- Epochs: 5
Training Results
Epoch	Train Accuracy	Validation Accuracy
1	80.95%	85.80%
2	86.38%	86.02%
3	87.43%	86.02%
4	88.25%	86.50%
5	88.95%	86.78%


Test Result
Test Accuracy: 85.91%
Test Loss: 0.4139
Compared with the baseline:
70.81% → 85.91%

Improvement:
+15.00 percentage points
This demonstrates the significant benefit of using pretrained visual representations.
7. Fine-Tuning MobileNetV2
After feature extraction, deeper layers of MobileNetV2 were selectively unfrozen.
The objective was to allow high-level visual features to adapt to CIFAR-10 while preserving the general visual representations learned from ImageNet.
Fine-Tuning Strategy
The MobileNetV2 backbone contained 154 layers.
The following strategy was used:
- Early MobileNetV2 layers → Frozen
- Last ~30 layers considered for fine-tuning
- Batch Normalization layers → Frozen
- 19 deeper non-BatchNorm layers → Trainable
- Final classification layer → Trainable
Learning Rate
A much smaller learning rate was used:
1e-5

A small learning rate helps make gradual adjustments to pretrained weights rather than making large updates that could destroy useful pretrained representations.
Training Configuration
- Optimizer: Adam
- Learning rate: 1e-5
- Batch size: 64
- Fine-tuning epochs: 5
Fine-Tuning Results
Epoch	Train Accuracy	Validation Accuracy
1	90.46%	87.94%
2	92.61%	88.38%
3	94.24%	88.68%
4	95.44%	89.24%
5	96.61%	89.20%


Test Result
Test Accuracy: 88.51%
Test Loss: 0.3619
Compared with feature extraction:
85.91% → 88.51%

Improvement:
+2.60 percentage points
8. Final Model Comparison
Model	Test Accuracy	Test Loss
CNN Baseline	70.81%	0.8460
MobileNetV2 Feature Extraction	85.91%	0.4139
MobileNetV2 Fine-Tuned	88.51%	0.3619


Overall Improvement
CNN Baseline
70.81%

        ↓ +15.00 percentage points

MobileNetV2 Feature Extraction
85.91%

        ↓ +2.60 percentage points

MobileNetV2 Fine-Tuning
88.51%

Overall improvement over the baseline:
+17.70 percentage points
 
9. Evaluation
The final fine-tuned model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Misclassified image analysis
10. Classification Report
Class	Precision	Recall	F1-score
Airplane	0.90	0.90	0.90
Automobile	0.95	0.95	0.95
Bird	0.90	0.85	0.87
Cat	0.75	0.82	0.78
Deer	0.84	0.88	0.86
Dog	0.86	0.79	0.82
Frog	0.90	0.91	0.91
Horse	0.93	0.88	0.90
Ship	0.91	0.94	0.93
Truck	0.93	0.94	0.93


Overall
- Accuracy: 88.51%
- Macro F1-score: 0.89
- Weighted F1-score: 0.89
11. Confusion Matrix
 
The largest individual confusion was:
Actual Dog → Predicted Cat = 137

while:
Actual Cat → Predicted Dog = 69

This shows an asymmetric confusion between the two visually similar classes.
Other notable confusions include:
Cat → Dog       = 69
Automobile → Truck = 36
Truck → Automobile = 32
Deer → Cat      = 30
Bird → Deer     = 43
Horse → Deer    = 41

12. Class-wise Performance
The model did not perform equally well across all classes.
Strongest Classes
Automobile
Precision: 0.95
Recall:    0.95
F1-score:  0.95

Truck
Precision: 0.93
Recall:    0.94
F1-score:  0.93

Ship
Precision: 0.91
Recall:    0.94
F1-score:  0.93

Weakest Classes
Cat
Precision: 0.75
Recall:    0.82
F1-score:  0.78

Dog
Precision: 0.86
Recall:    0.79
F1-score:  0.82

The lower recall for dogs indicates that a significant number of actual dog images were classified as other categories, with cat being the largest source of confusion.
13. Error Analysis
Misclassified images were visually inspected to understand the failure modes of the final model.
Cat vs Dog
Cat and dog images were among the most difficult examples.
The model frequently confused these classes because of their visual similarity.
The most significant confusion was:
Dog → Cat = 137 images

compared with:
Cat → Dog = 69 images

Deer vs Horse
Both classes contain four-legged animals with similar overall body structures and shapes.
Automobile vs Truck
Both categories contain road vehicles, resulting in some confusion between them.
Low Image Resolution
CIFAR-10 images are only:
32 × 32 pixels

At this resolution, fine-grained features such as:
- Facial structure
- Ears
- Fur patterns
- Body details
can be difficult to distinguish.
Background and Object Size
Some difficult examples contained:
- Small objects
- Complex backgrounds
- Unusual viewpoints
- Limited visual information
These factors can make classification difficult even for a strong pretrained model.
14. Fine-Tuning and Overfitting
Fine-tuning significantly improved the model's training performance, but the training-validation gap also increased.
At the end of fine-tuning:
Training Accuracy   = 96.61%
Validation Accuracy = 89.20%

The gap was approximately:
96.61% - 89.20% = 7.41 percentage points

This indicates that the model was beginning to fit the training data more strongly than the validation data.
The validation accuracy improved until approximately epoch 4 and then remained almost unchanged, while training accuracy continued to increase.
This indicates the beginning of overfitting and suggests that simply training for more epochs would not necessarily improve generalization.
  
15. Key Takeaways
Transfer Learning
ImageNet-pretrained features provided a major improvement over a CNN trained from scratch.
CNN Baseline
70.81%

MobileNetV2 Feature Extraction
85.91%

This produced a:
+15.00 percentage-point improvement
Fine-Tuning
Selective fine-tuning provided an additional improvement:
85.91% → 88.51%

This produced another:
+2.60 percentage-point improvement
Overall
The final MobileNetV2 model achieved:
88.51% test accuracy
compared with:
70.81% for the baseline CNN
Overall improvement:
+17.70 percentage points
Error Analysis
The main weaknesses were:
- Cat vs Dog confusion
- Deer vs Horse confusion
- Automobile vs Truck confusion
- Low-resolution and ambiguous images
16. Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Google Drive
- Git
- GitHub
17. Project Structure
CIFAR10-Transfer-Learning/
│
├── CIFAR_Transfer_Learning.ipynb
├── README.md
├── results.json
│
└── results/
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── fine_tuning_accuracy.png
    └── fine_tuning_loss.png

The trained .keras model files are maintained separately due to repository file-size considerations.
18. How to Run
1. Clone the repository
git clone https://github.com/Harshahere/CIFAR10-Transfer-Learning.git
cd CIFAR10-Transfer-Learning

2. Open the notebook
The project was developed using Google Colab.
Open:
CIFAR_Transfer_Learning.ipynb

in Google Colab.
3. Obtain the CIFAR-10 Dataset
Download the CIFAR-10 Python dataset and place/extract it according to the path specified in the notebook.
4. Run the Notebook
The notebook contains the complete workflow:
Dataset Loading
       ↓
Data Preprocessing
       ↓
Data Augmentation
       ↓
CNN Baseline
       ↓
MobileNetV2 Feature Extraction
       ↓
Selective Fine-Tuning
       ↓
Model Evaluation
       ↓
Error Analysis

19. Future Improvements
Potential improvements include:
- Experimenting with stronger data augmentation
- Hyperparameter tuning
- Learning-rate scheduling
- Early stopping
- Comparing additional pretrained architectures
- Testing different fine-tuning depths
- Experimenting with larger input resolutions
- Evaluating additional model architectures
- Deploying the trained model for inference
Author
G. Harsha Vardhan
B.Tech — Electrical Engineering
Indian Institute of Technology Tirupati
GitHub: Harshahere
Results at a Glance
Experiment	Test Accuracy
CNN from Scratch	70.81%
MobileNetV2 Feature Extraction	85.91%
MobileNetV2 Fine-Tuning	88.51%


Final Improvement
+17.70 percentage points
