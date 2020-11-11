# Suspicious-Behavior-Recognition
Detect human suspicious behavior, trained on Caviar dataset https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/.

I used pre-trained model ResNet50 model to apply transfer learning, where I didnt extact the last classification.
To classify the human action, we must add a RNN layer, i choose GRU(512 neurons) followed by one hidden layer with 1024 neurons.
Finally the output layer, in my case I decided to classify each frame if its contains a Suspicious action - 1 or not output 0.
I ran all the model in google colab.
![alt text](https://github.com/sodi16/Suspicious-Behavior-Recognition/blob/main/model.JPG)

# Run on Caviar Dataset
Download the trained model - **model.h5**

# Run on your own Dataset


# Results
![alt text](https://github.com/sodi16/Suspicious-Behavior-Recognition/blob/main/not_suspicious_frame.png)
![alt text](https://github.com/sodi16/Suspicious-Behavior-Recognition/blob/main/suspicious_frame.png)


