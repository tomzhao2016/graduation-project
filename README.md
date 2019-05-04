# PrivacyNet: An End-to-end Approach for Preserving Image Privacy
This repository provides a Keras implementation in addressing the Privacy-Utility Trade-off for large scale image data.
The private content is specified by a subset of training labels (private labels), and the public content is represented
by the rest of the training labels (public labels).
PrivacyNet learns an image-to-image translation, which desensitizes the original images, but ensures the image utility
and authenticity. On the desensitized images, the predictions on the private labels should be no more than random guesses.
On the contrary, when predicting public labels, the performance should be close to the original performance.
## Reversed MNIST data:
We reverse the MNIST data by changing the original white digits on the black background to black digits with white backgrounds.
Then, we set **whether the image is reversed** as the private content, and **the digit number** as the public content.
The desensitized images with different privacy levels are shown below.
![alt text](/privacynet_images/reversed_mnist_privacynet.png)
The private and public accuracy when testing on the desensitized images are:
![alt text](/privacynet_images/reversed_mnist_accuracy.png)

## CelebA human face data:
In this experiment, the private label is **the gender information**, and the public labels are **wearing heavy makeup,
wearing lipsticks, Arched Eyebrows, Wavy Hair, No beard**.
![alt text](/privacynet_images/new_celeba.png)
The private accuracy and the public accuracy when predicting on the private and pubic labels are:
<p align="center">
    ![alt text](/privacynet_images/celeba_accuracy.png)
</p>

## Getting Started
