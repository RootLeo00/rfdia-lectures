## 1.1 Supervised dataset
### 1. What are the train, val and test sets used for?
When we would like to use a model for a specific task, we usually go through several passages. The main ones are the training, that is the time when you want your model to learn the task, the validation, that is when you would like validate your training (for example, understanding if it is over-fitting), then you finally test your model, asking it to do the task it was meant to do. It is important that each step has its own part of the data that are not the same of the other steps.  For example, if you test your model with the same data of the training, then you would probably get good results, but it would be like cheating. The same goes for using the val set also as the test set.
In the end:
➤ Train set: used for training a given/chosen model on this data  
➤ Val set: used for model selection. (helps to tune the model). The test set should stay untouched so as not to falsify the generalization error.  
➤ Test set: used for the final model evaluation. This data should be kept away from the training process. During the final evaluation, the model sees this data for the first time.
### 2. What is the influence of the number of examples N ?
Generally the higher the number of available examples, the better the results you get. If one trains a complex model with too little data, the model parameters will be overly adapted to the few training examples thus leading to over-fitting ( ⇒ see bias-variance trade-off for more info on over-fitting and model choice). Usually there is no downside on training a model with too many data, but only that it would be much more time consuming.

## 1.2 Network architcture (forward)
### 3. Why is it important to add activation functions between linear transformations?
First, it makes the combination of transformations (linear + activation) non-linear, thus justifying applying several transformations in a row. 
This is because the composition of (multivariate) linear function would still be linear. Therefore, without the activation functions, the network would only be able to approximate combinations of linear functions. For more information, see the universal approximation theorem (one of the assumption is having a non-linearity σ)

Second, it allows to choose an output interval different from R^ny

### 4. What are the sizes nx, nh, ny in the figure 1? In practice, how are these sizes chosen?
![[Pasted image 20231012154658.png]]
In theory, nx is determined by the size of the input data, therefore (apart from the choice of how to encode the input data) we do not have to choose this value. nh should be sufficiently large so as not to loose salient information in the process of passing data through the network but sufficiently small such that the NN can extract the most salient features. The smaller nh, the tighter the bottleneck of our neural network. ny is determined by the purpose of our NN. For classification, for example, ny depends on the number of output classes.
Specifically, in figure 1 the size of nx is 2, ny is 2 and nh is 4.

### 5. What do the vectors ˆy and y represent? What is the difference between these two quantities?

y are the true class labels (ground truth), and yhat are the predicted class labels/class probabilities as output of the network. The model aspires to get y=yhat as much as possible

### 6. Why use a SoftMax function as the output activation function?
The softmax function normalizes the output probabilities such that they add up to one. It is used for multi-class classifcation (usually with more than 2 classes, otherwise it is prefered to use the sigmoid function). Thus the softmax function can be seen as a generalization of the sigmoid function which was used to represent a probability distribution over a binary variable. Note that we do not usually use the argmax() because with that we operate in a too strong manner on the output data, flattening the output vector in one where 1 correspond to the predicted label and 0 everything else. Thus, softmax is called like so because it keeps more information on the weights: it puts the most weight on class 1 and less weight on the other classes, keeping the "error".

### 7. Write the mathematical equations allowing to perform the forward pass of the neural network, i.e. allowing to successively produce ˜h, h, ˜y and ˆy starting at x.

![[Pasted image 20231012160845.png]]
Instead of ReLu, we can use any other activation function, such as the tanh:
ĥ= W_h*x + b_h
h = tanh(ĥ)
ỹ= W_y*h + b_y
ŷ = SoftMax(ỹ)

## 1.3 Loss function
### 8. During training, we try to minimize the loss function. For cross entropy and squared error, how must the ˆyi vary to decrease the global loss function L?
For cross-entropy loss, you adjust ˆyi to align with true class probabilities,  while for squared error loss, you adjust ˆyi to minimize the squared differences between predictions and true values.
Looking at the mathematical formula of the cross-entropy that is:
![[Pasted image 20231012161624.png]]
we can find two limit cases:
- if yi is 1 (thus, the correct label), then log(yihat) should be >> 1, so yihat should be >1, because we want to decrease as much as we can the loss function overall (remember that there is a minus)
- if yi is 0 (thus, the wrong label), then log(yihat) could be any number, so we prefere to assign yihat to 0 (as possible). Note that, although yi is 0 and multiplies log(yihat), remember that yhat is a vector that should have one label near 1 and the others near to 0, so we don't want here that yihat is the chosen 1. Note again that in the limit case where yihat is 0, we would in theory get 0 * -inf, but since the log function is slower than the y function, then the y function wins on the log function so we get 0 as output and not -inf. Also usually in practice we sum y^i to an arbitrary epsilon constant (let's say 10^-10) in order to not worry about this limit case.

In a similar manner, looking at the mathematical formula of the mean squared error, that is:
![[Pasted image 20231012162344.png]]
- if yi is 1 (thus, the correct label), then yihat should be 1
- if yi is 0 (thus, the wrong label), then yihat should be 0. Note that there is the power by 2, so any negative number here would be then converted to the absolute value, therefore it would increase the loss (which we don't want)

In general:
- if yi is 1 (thus, the correct label), then yihat should be as much equal to 1 as we can
- f yi is 0 (thus, the wrong label), then yihat should be as much equal to 0 as we can

### 9. **How are these functions better suited to classification or regression tasks?**

The MSE loss is better suited to regression tasks (--> see χ2methods).
In classification problems since we have classes as outputs we want our loss to penalize a lot an error. The MSE doesn't penalize the loss as much as it should do. Comparing the two formulas:
![[Pasted image 20231012161624.png]]
![[Pasted image 20231012162344.png]]
Let's write the formula in the case of a single element of the y / yhat vector:
l(y,y^) = - y * log y^
l(y,y^) = y - yhat
For the sake of simplicity, let's consider the case of y=1:
l(y,y^) = - log y^
l(y,y^) =  (1 - yhat)^2
If we plot these 2 function we get the following:
![[Pasted image 20231012175551.png]]
If we get yhat=0.1, then we get these points:
![[Pasted image 20231012175306.png]]
Here the CE function is higher then the penalty will be higher.

Let's write the formula in the case of a single element of the y / yhat vector in the case that 1. **Cross-Entropy Loss for y=0**:
    
The formula for CE loss for y=0 is: l(y,y^)=−y⋅log⁡(y^)l(y,y^​)=−y⋅log(y^​)
    
In this case, where y=0: l(0,y^)=−0⋅log⁡(y^)=0l(0,y^​)=−0⋅log(y^​)=0
    
When the true label y is 0, the CE loss is always 0, regardless of the predicted value y^y^​. This is because the loss only measures the divergence when the true label is 1.
    
2. **Mean Squared Error Loss for y=0**:
    
    The formula for MSE loss for y=0 is: l(y,y^)=(y−y^)2l(y,y^​)=(y−y^​)2
    
    In this case, where y=0: l(0,y^)=(0−y^)2=y^2l(0,y^​)=(0−y^​)2=y^​2
    
    When the true label y is 0, the MSE loss is directly proportional to the square of the predicted value y^y^​. This means that the MSE loss will increase as y^y^​ moves away from 0.
    

So, in contrast to cross-entropy loss, which doesn't penalize the model when the true label is 0, the mean squared error loss will penalize the model by the square of the predicted value. As you mentioned, the choice of loss function depends on the nature of the problem and the desired behavior. Cross-entropy is better suited for classification problems because it penalizes incorrect class predictions, especially when the true label is 1, while MSE is designed for regression tasks and may not be as appropriate for classification problems.

Why don't we use the CE for a regression problem? This is because in a regression problem we are dealing with continuous values. So even if our regression model gives an answer a little bit different from the ground truth it should not cost the model too much. 

[reference](https://www.google.com/search?client=ubuntu-sn&hs=Lj1&sca_esv=572890011&channel=fs&sxsrf=AM9HkKnhQ3riOlpxj7AsMK1G-mbk8qtbEg:1697123750871&q=why+cross+entropy+for+classification&tbm=vid&source=lnms&sa=X&ved=2ahUKEwivw73S5vCBAxW6VKQEHa_-BlEQ0pQJegQICBAB&biw=1542&bih=807&dpr=1.2#fpstate=ive&vld=cid:d1ae8bea,vid:gIx974WtVb4,st:0)
## 1.4 Optimization algorithm
### 10. What seem to be the advantages and disadvantages of the various variants of gradient descent between the classic, mini-batch stochastic and online stochastic versions? Which one seems the most reasonable to use in the general case?

it computes the exact gradient of the loss function using the entire training dataset in each iteration therefore its more accurate and gives smoother convergence trajectory.
Calculating the hole gradient loss is a very heavy computation, that's why usually it is more preferred to use the mini-batch stochastic. In fact, in the stochastic gradient descent we divide the train set in small parts called mini-batches and we compute the gradient on them. Since we are computing the gradient descent in a much smaller number of data we gain a much more faster computation than the classical approach.
it updates the model’s parameters based on a single training example at a time, which is computationally very efficient. A usage example is when you have a stream of data. Since it doesn't provide many data for the computation, the gradient descent path in the training could be unstable. In the general, mini-batch SGD is often considered the most reasonable  
choice for training neural networks and machine learning models. It combines the advantages of both classic GD and online SGD while mitigating  some of their disadvantages.

### 11. What is the influence of the learning rate η on learning?

The learning rate η determines the incremental rate with which the gradient descend algorithm is updated. A large eta will take big steps and converge faster to a sensible point but might overshoot and not find ... (finetuning).  
On the other hand, a learning rate that is too small will get stuck in local minima or not converge fast enough. To improve the performance, different update algorithms have been developed that take into account e.g. momentum (Adam), or the future state (Ada).

### 12. **Compare the complexity (depending on the number of layers in the network) of calculating the gradients of the loss with respect to the parameters, using the naive approach and the backpropagation algorithm.**

**Naive Approach**:
In the naive approach, you would directly compute the gradients of the loss with respect to the parameters using the definition of the derivative. 
- For each parameter, you perform a forward pass through the entire network, which has a computational cost of O(N) for one parameter.
- You need to compute the derivative with respect to each parameter individually, resulting in O(N) operations for each parameter.

So, the total computational complexity for the naive approach is O(N^2), as you're repeating this process for each parameter. This becomes impractical as the number of parameters increases.

**Backpropagation Algorithm**:

Backpropagation is a much more efficient and scalable approach for calculating gradients in neural networks. It leverages the chain rule of calculus to compute gradients layer by layer, starting from the output layer and moving backward through the network. The algorithm computes the gradients efficiently without redundant calculations. 
- You perform one forward pass through the network, which has a computational cost of O(N) where N is the total number of parameters in the network.
- You then perform a backward pass, which computes gradients layer by layer. The backward pass is roughly O(L) because you compute gradients for each layer in sequence.
- Within each layer, the computation of gradients is O(N), as it depends on the number of parameters in that layer.

The total computational complexity of backpropagation is O(L * N), which scales linearly with the number of layers and parameters in the network. This is significantly more practical for deep neural networks.

### 13. **What criteria must the network architecture meet to allow such an optimization procedure ?**

**Layer Connectivity**: The architecture should have a clear and sequential structure with well-defined layer connections. Each layer should connect to the next layer, and information should flow forward without skipping or looping connections.

**Feedforward Structure**: Neural networks typically have a feedforward structure where data flows from input layers to output layers without feedback loops. Recurrent neural networks (RNNs) are an exception, but they have a different training procedure.

**Differentiability**: The network's activation functions and loss function must be differentiable. This is essential for gradient-based optimization methods like backpropagation to calculate gradients and update the network's parameters.

### 14. The function SoftMax and the loss of cross-entropy are often used together and their gradient is very simple. Show that the loss can be simplified by

![[Pasted image 20231012220849.png]]
![[Pasted image 20231012222513.png]]
#TODO change the minus 

### 15. Write the gradient of the loss (cross-entropy ) relative to the intermediate output ˜y

![[Pasted image 20231012222543.png]]
![[Pasted image 20231027212148.png]]

### 16. Using the backpropagation, write the gradient of the loss with respect to the weights of the output layer ∇Wy `. Note that writing this gradient uses ∇˜y`. Do the same for ∇by `.
See: https://cs229.stanford.edu/main_notes.pdf
https://medium.com/@ilmunabid/beginners-guide-to-finding-gradient-derivative-of-log-loss-by-hand-detailed-steps-74a6cacfe5cf
https://towardsdatascience.com/calculating-gradient-descent-manually-6d9bee09aa0b
https://www.youtube.com/watch?v=lZrIPRnoGQQ
https://www.youtube.com/watch?v=tIpKfDc295M

17.  Compute other gradients : ∇˜h`, ∇Wh `, ∇bh `
![[1ab-answer16-17_231021_142243.pdf]]


#TODO check on other pdf