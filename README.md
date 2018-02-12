# Semantic Segmentation
## Goal

The goal of this project is to implement a fully convolutional network in order to classify road segments from images. The baseis for this netowrk is the VGG-16 image classifier architecture, which can be downloaded as pretrained model. This pretrained model allows for a shorter computational time, when training the fully convolutional network. Similar to transfer learning, the learned weights can give a good start for this classification problem.

## Approach

In order to not start out in the blue, the proposed network architecture from the course was used. 

![Architecture](.pics/architecture.png)

### Regularization

Regularization was introduced in order to keep the weights low. Regularization weights can be added to the loss function, this way, as soon as the weights of the individual layers get to high, they are taxed with a high cost and thus abandoned by the optimized. The overall cost computes as follows

COST = Cross Entropy Loss + 0.004 * SUM(Regularization Weights)

This cost is minimized using an Adam Optimizer with a learning rate of 0.0005 and a keep probability of 0.5.

## Results

The first training results reached a loss of 0.5, which proved to be nowhere close to an acceptable result. Increasing the number of epochs and decreasing the learning rate has proven to be a good approach to reach very low loss values. The final decision to take only 25 epochs was made mostly due to computational time, although the trend in loss values was still going down.

### Example Pictures

The following pictures show the result of the network.

![Example 1](.pics/ex1.png)
![Example 2](.pics/ex2.png)
![Example 3](.pics/ex3.png)
![Example 4](.pics/ex4.png)