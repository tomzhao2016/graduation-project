# PrivacyNet: An End-to-end Approach for Preserving Image Privacy
This repository provides a Keras implementation in addressing the Privacy-Utility Trade-off for large scale image data.
The private content is specified by a subset of training labels (private labels), and the public content is represented
by the rest of the training labels (public labels).
PrivacyNet learns an image-to-image translation, which desensitizes the original images, but ensures the image utility
and authenticity. On the desensitized images, the predictions on the private labels should be no more than random guesses.
On the contrary, when predicting public labels, the performance should be close to the original performance.
## Reversed MNIST data:
We reverse the MNIST data by changing the original white digits on the black background to black digits with white backgrounds.
Then, we set whether the image is reversed as the private content, and the digit number as the public content.
The desensitized images with different privacy levels are shown below.
<img src="https://github.com/tomzhao2016/graduation-project/privacynet_images/reversed_mnist_privacynet.pdf" width="1000px"/>