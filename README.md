# Suspicious-Behavior-Recognition
Detect human suspicious behavior, trained on Caviar dataset https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/, model in Keras.
* Convert video to frames (5fps) for all videos
* Extract from my groundtruth.xml files, my labels wich are the human activity in each frame
* Identify which of them are suspicious and which are not
* Prepare Data, augment it, save it for the model
* Run my model (on colab) and save his weights

#### Model
I used pre-trained model *ResNet50* model to apply transfer learning, where I didnt extact the last classification.
To classify the human action, we must add a *RNN* layer, i choose *GRU* (512 neurons) followed by one hidden layer with 1024 neurons.
Finally the output layer, in my case I decided to classify each frame if its contains a suspicious action  - 1 or not output - 0.


## Run on Caviar Dataset
Download the trained model - **model.h5**

## Run on your own Dataset
If you want to use your own dataset, you have to have to follow all the steps I wrote above.
Only when you have all your frames saved and their 

## Results
![alt text](https://github.com/sodi16/Suspicious-Behavior-Recognition/blob/main/not_suspicious_frame.png)
![alt text](https://github.com/sodi16/Suspicious-Behavior-Recognition/blob/main/suspicious_frame.png)


