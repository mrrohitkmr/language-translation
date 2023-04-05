# language-translation

This code is a Convolutional Neural Network (CNN) model for image classification using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The objective of the model is to correctly classify each image into its corresponding class.

# Model architecture
The CNN model consists of four convolutional layers with ReLU activation functions, max-pooling layers, and dropout layers for regularization. The first three convolutional layers have 32, 48, and 80 filters respectively with a 3x3 kernel size and same padding. The fourth convolutional layer has 128 filters with a 3x3 kernel size and same padding. The last two layers are fully connected layers, with 500 and 10 units respectively. The output layer has a softmax activation function to output the probabilities of the 10 classes.

The model uses the Adam optimizer with a learning rate of 0.0001 and a categorical cross-entropy loss function. It is trained for 100 epochs with a batch size of 32. The input images are normalized by dividing each pixel by 255.

The model also implements data augmentation using Keras' ImageDataGenerator. This helps to increase the number of training examples by generating new examples through random transformations such as rotations, shifts, and flips. This reduces the risk of overfitting and improves the model's generalization ability.

The TensorBoard callback is used to visualize the training and validation loss and accuracy during training.

# Evaluation
Finally, the model is evaluated on the test set using the evaluate() function, which outputs the test loss and accuracy.

Overall, this CNN model achieves a test accuracy of approximately 73%. There is still room for improvement, and one possible way to improve the model is to add more convolutional layers or increase the number of filters in the existing layers. Additionally, the hyperparameters such as learning rate, batch size, and number of epochs can also be tuned for better performance.
