# Suspicious-Behavior-Recognition
Detect human suspicious behavior, trained on Caviar dataset https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/.

  **Caviar_dataset.py** Create my caviar object, that will store my data.    
  **create_data.py** By using the caviar object I converted video to frames, pick the desired number of frames per seconds (5fps in my code),   extract labels from my groudntruth.xml file.  
  I defined which of them are suspicious labels = 1, and which labels are not considered as suspicious and assign them value 0.   
   Also keep the possibility to display video with their labels, like the bounding boxes and the description.  
  Repeat it for all the videos of a same category, and save them as data and labels.  
	**Model_colab.ipynb** regroup all my video data and labels together, augment my desired video data, and pass them in my model.

### Model
I used pre-trained model *ResNet50* model to apply transfer learning, where I didnt extact the last classification.
To classify the human action, we must add a *RNN* layer, i choose *GRU* (512 neurons) followed by one hidden layer with 1024 neurons.
Finally the output layer, I decided to classify each frame if its contains a suspicious action - 1 or not output - 0.

## Results
![alt text](https://github.com/sodi16/Suspicious-Behavior-Recognition/blob/main/not_suspicious_frame.png)
![alt text](https://github.com/sodi16/Suspicious-Behavior-Recognition/blob/main/suspicious_frame.png)


