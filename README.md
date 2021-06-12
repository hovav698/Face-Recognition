This repository contains pytorch implementation of face recognition algorithm. 
I will first briefly go through the theory and explain the basic implementation.

**Dataset description, and what is the goal?**

I use the Yale face dataset, it consists of 166 face images of 15 different people. The goal is to build model that receive two images of person face, and outputs wheather the face belong to the same person or not.

**How the model works?**

The model that I implements for solving the problem is called "Siamese neural network". The network receives 2 images, extracts the image feautres, and outputs the oclidean distance between the embeddings of the images. For the lost computation I will use a contrastive lost function - for images of identical people the the model will try to minimzed the distance between the two feature vectors, and for images of different people the model will try to maximize the distance for the two vectors.
In the end of the training process, the model will learn to output bigger distance to pair of faces that belong to different people, and smaller distance for images that belong to the same person. See the model architecture in the following picture:

![Alt Text](https://lh4.googleusercontent.com/s-5BhW6gHkOTAfrGoJOCArAt1JPmpp1XCZyFaIqvvExUyIrfDxQb4_4SmtqWlV9qjBFKWnhB8CCCRe5RR5p0v7p64GAeMlEQAXOoO21gmWJ9CMoPtKMYiKMT4qvzY-F9F7XT8IwSIUcIY_Erfg)

**Data Preperation**

The dataset contains all the images of the different faces, but the model excpect two images of faces as input, therefor I will create a list of indexes for all the possible pairs of images. The model will get the images of those index pair, and the target for the loss computation will be 1 if the image pair belong to the same person, and 0 otherwise.
I will use data genertor and split it into batches, since I don't want to save the images for every combination to the disk, since it won't be efficient and will take too much space.


**Deciding About Threshold**

After the model has learned the differences between embedding of similar and different pair of faces, I will need to determine what is the distance that best seperates between images pairs of same and different faces. I will do it by look at the distance distribution of images pairs of the same face vs the different faces.

![image](https://user-images.githubusercontent.com/71300410/121767966-40d80880-cb64-11eb-8c77-52abb773900c.png)

From the histogram it looks that the distance that best seperates the data is around 0.5. For deciding about the best threshold, we need to decide if we care more about correctly predicting similar faces, in that case we will choose treshold that gives high sensetivity, however it will lower the specificity. If we care more about correctly predicted non-similar face, then we will choose treshold that gives high specificity, however it will lower the senitivity.
The threshold I choose for the project purpose was 0.7.

**Final Step**

So the final step is to input the faces into the model, if the model outputs distance bigger than the threshold, I will define it to predict "face mismatch", otherwise precit "face match":

![image](https://user-images.githubusercontent.com/71300410/121768216-bbedee80-cb65-11eb-995f-721bf0dba100.png)









