### 1. Considering a single convolution filter of padding p, stride s and kernel size k, for an input of size x × y × z what will be the output size ? How much weight is there to learn ? How much weight would it have taken to learn if a fully-connected layer were to produce an output of the same size ?
The output size of a convolutional layer can be calculated using the following formula:


1. **Output Size**:
    
    For a single convolution filter with padding pp, stride s, and kernel size k applied to an input of size x×y×z, the output size will be:
    
    x' = [(x + 2p - k)/s] + 1
	y' = [(y + 2p - k)/s] + 1
    
    The output depth is determined by the number of filters in the layer. Each filter produces one channel in the output.
    
2. **Number of Weights to Learn**:
    
    The number of weights to learn in a convolutional layer is determined by the kernel size and the number of input channels. For each filter, you have kxkxc weights, where k is the kernel size, and c is the number of input channels.
    
    So, the total number of weights for one filter is k×k×c, and if you have N filters in the layer, you'll have N×k×k×c weights to learn.
    
3. **Comparison to Fully-Connected Layer**:
    
    If a fully-connected layer were to produce an output of the same size as the convolutional layer, you would have to connect each input neuron to each output neuron. In this case, the number of weights in the fully-connected layer would be x×y×z for each output neuron.

### 2. What are the advantages of convolution over fully-connected layers ? What is its main limit ?
There are 2 main problems of the brute force fully connection of images:
- in the MLP where you have fully connected layers only the representation of the input is not stable with respect to some variation. For example, what we expect for good visual representation is that if you have small variations/deformations in the input, then you should get a representation that is similar to the original input. The information should stay the same. On the other side, if you have a large variation, you should get a very dissimilar representation from the original one. This kind of stability is necessessary expecially when you want to extract a semantic freature of the input (such in a semantic segmentation task). In the case o fully connected models we do not have this stability: small changes in the input can result in significant changes in the network's output. For example, if you translate an image slightly, a fully connected network may not recognize the same pattern as in the original position. 
For example, in the figure below, you see that the green dots and the red dots are closed to each other, and you if you flatten the pixeled-image you would get the vestor of black and white input that is shown. If we shift the black pixels of 2 positions in the right, we would get a large number of input neuron that would switch from white to black and viceversa (here, about 150), that leads to a very different image representation. 
![[Pasted image 20231013112331.png]]
What's even worse is that if you take an input image and you randomly permute its pixels and you train it in a fully connected nn, you would get similar results as if you trained it with the non-permuted original image! This means that the topology of the image is completely ignored.

- There is a larger number of weights compared to the convolutional layer, making it computationally expensive and prone to overfitting, especially when the input size is large.
	Example:
	If you have an image that is 1000x1000 then after flattening it you would get an input vector of 10^6 (11 million) parameters. If then you choose again to have an hidden layer of the same size of the previous layer, then you have a hidden layer of 10^6 parameters and each neuron is fully connected to every previous neuron, meaning that you get 10^6x10^6 number of parameters in all that hidden layer. This is a lot to store in normal computers and it is just one layer (probelm of scalability)!
	
	![[Pasted image 20231013113239.png]]

On the opposite side, there are two main ideas for the convolutional layers:
- first one is to make local connections instead of global connections: each hidden layer of the NN is not connected to all of the neurons of the next one, but just to some of them. This is called sparse connectivity (vs full connectivity), and the the weights connected to a local patch of the neurons are called filter or kernel. Another way of saying that, is that the convolutional layers are sensitive to a small sub-region of the input space, called receptive field (analogy with biology).
Example:
	If you have an image that is 1000x1000 then after flattening it you would get an input vector of 10^6 (11 million) parameters. If then you choose again to have an hidden layer of the same size of the previous layer, then you have a hidden layer of 10^6. If the hidden layer is a convolutional layer, we decide to connect a small number of neurons in the input layer to each neuron of the hidden layer. Since we have an image as input, we arbitrarly decide to connect small patches of the image that are big 10x10. This means that each hidden neuron will be connected to 10^2 input-neurons. Therefore, the total number of parameters f the first hidden layer is 10^6x10^2=10^8, that is way less then the results we got previously with the fully connected nn. 
![[Pasted image 20231013113620.png]]
- second, you have shared weights. It means that you use the same set of weights (also known as a kernel or filter) across multiple spatial locations in an input feature map. This means that instead of having a separate set of weights for each location in the input, a single set of weights is applied across the entire input. This is crucial in order to achieve the stability that we cannot afford in the fully connected layers, such as equivariant to translation: if you shift your original image and apply convolution, you would get the same results of the original image but with the shifting. This means that objects or patterns can appear at different positions in an image, but their identity remains the same. By sharing weights, CNNs can recognize the same feature regardless of its position in the input (you have string spatial information). On the other hand, contrary of what said for the FC, if you permute the input, you would get a totally different output of the convolutional layer respect to the output with the non-permuted input.

Why are this type of layers called convolutional layers?
If you satisfy these two constraints, it is equivalent to compute the output layer as a convolution of the input layer (see slide 16).

When you train your model, your goal is to learn the kernels. So in CNN the weights are the kernels. 

When you apply one convolution with 1 specific kernel to an image (or feature map), you get 1 image (or feature map) as result. Usually, we like to apply multiple kernel in order to ouput several feature map and get a richer description (= get more parameters). If between an input layer and a following hidden layer you apply n convolution, you would get n outputs (see image slides 18-19).
Using multiple filters, you would get not many weights (for the constraint of sharing them), but you would get many neurons and this would get to memory issues. 

When you apply the convolution to a certain input, you want to have the same dimension for the output you have to configure the padding and the stride. 

In some tasks, expecially in classification, you want your convolutional layer not only to be equivariant to translation operations but invariant. The main difference between invariance and equivariant is that the equivariance allows the network to generalise edge, texture, shape detection in different locations, while the invariance allows precise location of the detected features to matter less. To get more local invariance we can add some pooling operations in the network. See question 3 for more details.


### 3.  Why do we use spatial pooling ?

In some tasks, expecially in classification, you want your convolutional layer not only to be equivariant to translation operations but invariant. The main difference between invariance and equivariant is that the equivariance allows the network to generalise edge, texture, shape detection in different locations, while the invariance allows precise location of the detected features to matter less. To get more local invariance we can add some pooling operations in the network. The pooling aggregates the value that wehave in the input feature map to sum up the info that we have in local regions. There are several way to pool the value (max, average, ..). In the image below is shown that after you translate your input image, if you apply convolution you get a shifted result (since convolutional layers are equivariant), and to fix that shifts you can apply a pooling operation that will give you the same result as if you computed the same pipeline with a non shifted input image.

Getting into the details of the pooling operations, you apply the pooling to each channel of the input. You can tweak several parameteers of the poolin, such as stride or padding. Usually after pooling we don't want to get an output image that has the same size of the input image, so we choose a stride that is >1, and so we downscale the image. 

### 4. Suppose we try to compute the output of a classical convolutional network (for example the one in Figure 2) for an input image larger than the initially planned size (224 × 224 in the example). Can we (without modifying the image) use all or part of the layers of the network on this image ?

No, you cannot use **all** of the part of the layers. Given the VGG16 network as example, if you give it an input image that is larger than 224x224 (let's say you give an image 400x400), every convolutional layer and max pooling (for example the output of the first convolutional layer would be 400x400x64) but then it won't fit in the fully connected layer. This is because when you design your model you have to fix the size of the  input and output of the fully connected layer :
```python
nn.Linear(input_size, output_size)
```
Instead, for convolutional layer, you don't specify the input or output size, but just the number of channels (= number of filters) in input and output and the size of the kernel:
```python
nn.Conv2d(in_channels, out_channels, kernel_size)
```
This is also the case for Max Pooling operations.
### 5. Show that we can analyze fully-connected layers as particular convolutions.

Mathematically, the operation of a fully-connected layer and a convolutional layer with a kernel size equal to the input size is the same. In both cases, each output unit computes a weighted sum of all the input values.

- In a fully-connected layer, if we consider the weight matrix W as the convolutional kernel, and the input as a 2D or 3D grid, then the output is given by O=W⋅I, where O is the output, W is the weight matrix, and I is the input.
    
- In a convolutional layer with a kernel size matching the input size, the convolution operation is represented as O=I∗K, where O is the output, I is the input, and K is the convolutional kernel. If the size of K is the same as the size of W, zero padding and an arbitrary stride, this is equivalent to the fully-connected layer operation.
Another method would be do a convolution with 1x1 kernels.
### 6. Suppose that we therefore replace fully-connected by their equivalent in convolutions, answer again the question 4. If we can calculate the output, what is its shape and interest ?

![[Pasted image 20231013150005.png]]
We want to replace the FC layers with the equivalent convolutional layer. 
The FC6 layer with c=4096 that is looking at some input volume of size 7×7×512 can be equivalently expressed as a CONV layer with k=7,P=0,S=1,c=4096, where k is the kernel size, p is the padding, s is the stride and c is the number of filters. In other words, we are setting the filter size to be exactly the size of the input volume, and hence the output will simply be 1×1×4096 since only a single depth column “fits” across the input volume, giving identical result as the initial FC layer.
IMAGE
Then the fc7 layer with again c=4096 will have input volume of size 1×1×4096. The equivalent convolutional layer would have k=1, p=0, s=1, c=4096.
IMAGE
Then the fc7 layer with again c=1000 (that is the number of classes of imagenet) will have input volume of size 1×1×4096. The equivalent convolutional layer would have k=1, p=0, s=1, c=1000.
IMAGE

Overall, this would be our new architecture:
IMAGE

If all the layers are convolutions, the neural network can be applied to images of any size. The size of the network's output depends on the size of the input images it is applied to. However, if the network is applied to smaller images, the output will be empty because the images are not large enough to apply the learned filter. If the images are larger, the output of the network will no longer be a single number but a 2D array. ( #TODO : revision )

[source](https://cs231n.github.io/convolutional-networks/#convert)
### 7. We call the receptive field of a neuron the set of pixels of the image on which the output of this neuron depends. What are the sizes of the receptive fields of the neurons of the first and second convolutional layers ? Can you imagine what happens to the deeper layers ? How to interpret it ?

![[Pasted image 20231013153617.png]]
[reference](https://cs231n.github.io/convolutional-networks/#conv)
- As you move deeper into the network, the receptive field size continues to increase. This is primarily due to the stacking of convolutional layers and pooling layers with larger windows or strides.
- Deeper layers have progressively larger receptive fields, allowing them to capture more global information and high-level features in the input image.
- This increase in receptive field size is a fundamental aspect of hierarchical feature learning in deep CNNs. Features learned in deeper layers tend to represent more complex and abstract patterns.

On the output of the first convolutional layer, the size of the pixel's receptive field is equal to the size of the convolution filter. By applying a second convolution immediately, the size of the pixel's receptive field becomes (k2-1) * s1 + k1.

The size of the receptive field increases with depth, which means that the early layers have low-level features, and the later layers have higher-level features.

### 8. For convolutions, we want to keep the same spatial dimensions at the output as at the input. What padding and stride values are needed ?
p=2, s=1

Notice that, to maintain the size of the images when applying a convolution, a stride of 1 is used, along with padding based on the size of the kernel 'k':

- For an odd 'k', a padding of size (k - 1) / 2 is appropriate.
- For an even 'k,' you should also use a padding of size (k - 1) / 2, but this results in a non-integer padding size. That's why, in practice, odd-sized kernels are used.

### 9. For max poolings, we want to reduce the spatial dimensions by a factor of 2. What padding and stride values are needed ?
p=0, s=2

### 10.  For each layer, indicate the output size and the number of weights to learn. Comment on this repartition.
You can see it in the code, or otherwise here:
10)
Layer | Size of the output | Number of weights
entrée 32*32*3 0
conv1 32*32*32 5*5*3*32 = 2400
pool1 16*16*32 0
conv2 16*16*64 5*5*32*64 = 51200 
pool2 8*8*64 0
conv3 8*8*64 5*5*64*64 = 102400
pool3 4*4*64 0
fc4 1000 4*4*64*1000 = 1024000
fc5 10 1000*10 = 10000
Total number of parameters: 1 190 000

( #TODO : revision)

### 11. What is the total number of weights to learn ? Compare that to the number of examples.
To calculate the total number of weights to learn in a neural network, you need to sum up the weights in all the layers of the network.
1. Convolutional Layers:
    - For each convolutional layer, you need to consider the number of input channels, the number of output channels (filters), the size of each filter, and whether or not there is a bias term. The formula for calculating the number of weights in a convolutional layer is:
        Number of Weights = (Input Channels * Output Channels * Filter Height * Filter Width) + (Output Channels if there's a bias term)
2. Fully Connected Layers:
    - For each fully connected (linear) layer, you need to consider the number of input neurons and the number of output neurons, including bias terms. The formula for calculating the number of weights in a fully connected layer is:
        Number of Weights = (Input Neurons * Output Neurons) + (Output Neurons if there's a bias term)
3. Sum up the number of weights for all layers to find the total number of weights in your network.

Method by hand:
1. Convolutional Layer 1:
    - Input Channels: `3` (assuming color images)
    - Output Channels (Filters): `32`
    - Filter Height: `5`
    - Filter Width: `5`
    - Bias Term: Yes
    - Number of Weights = `(3 * 32 * 5 * 5) + 32 = 2432`
2. Convolutional Layer 2:
    - Input Channels: `32` (output from the previous layer)
    - Output Channels (Filters): `64`
    - Filter Height: `5`
    - Filter Width: `5`
    - Bias Term: Yes
    - Number of Weights = `(32 * 64 * 5 * 5) + 64 = 51264`
3. Convolutional Layer 3:
    - Input Channels: `64` (output from the previous layer)
    - Output Channels (Filters): `64`
    - Filter Height: `5`
    - Filter Width: `5`
    - Bias Term: Yes
    - Number of Weights = `(64 * 64 * 5 * 5) + 64 = 102464`
4. Fully Connected Layer 4:
    - Input Neurons: `64 * 4 * 4` (output from the last convolutional layer, which is 64 channels, each of size 4x4)
    - Output Neurons: `1000`
    - Bias Term: Yes
    - Number of Weights = `(64 * 4 * 4 * 1000) + 1000 = 1025000`
5. Fully Connected Layer 5:
    - Input Neurons: `1000` (output from the previous fully connected layer)
    - Output Neurons: `10` (assuming this is a classification layer)
    - Bias Term: Yes
    - Number of Weights = `(1000 * 10) + 10 = 10010`

Now, let's sum up the number of weights from each layer:
- Convolutional Layers: `2432 + 51264 + 102464 = 154160` weights
- Fully Connected Layers: `1025000 + 10010 = 1035010` weights

Total Number of Weights in the `ConvNet2` model: `154160 + 1035010 = 1199170` weights.

Method coded:
look at count_parameters() function in project1cd2.py:
```python
def count_parameters(self):
	total_params = 0
	for param in self.parameters():
		total_params += param.numel()
	return total_params
```

Comparing the number of the weights to be learned (1190000) with the number of the images in the training dataset (50000), we think that the number of images is too small to learn such number of weights and this could lead to underfitting or overfitting.
### 12. Compare the number of parameters to learn with that of the BoW and SVM approach.
Certainly, here's a more detailed and extended version:

In the previously employed Bag of Words (BoW) approach coupled with Support Vector Machine (SVM) classification, utilizing a dictionary comprising 1000 Scale-Invariant Feature Transform (SIFT) descriptors, the total number of parameters to be learned was approximately 128,000. This method involved a series of hyperparameters that needed to be carefully configured, including the size of the learned dictionary, the tuning of the 'C' constant in the SVM classifier, and the specific type of SIFT descriptors used.

In contrast, when employing a convolutional neural network (CNN), we encounter a substantially higher number of parameters to train. To be precise, the CNN demands learning tenfold more weights compared to the BoW+SVM approach. Additionally, the neural architecture introduces a set of hyperparameters to consider, but it offers a unique advantage. The CNN operates on an end-to-end learning paradigm, which enables it to automatically acquire and adapt the feature extraction process.

This means that, with a CNN, the model learns not only how to classify images but also how to extract and represent the most relevant features from the data, whereas the BoW+SVM approach relies on predefined features and requires fine-tuning of its parameters to achieve optimal performance. The increase in parameter count in the CNN is a trade-off for this added capacity to learn a broader range of features directly from the data.

( #TODO : revision)

### 13. Read and test the code provided. You can start the training with this command : main (batch_size, lr, epochs, cuda = True)
...
### 14.  In the provided code, what is the major difference between the way to calculate loss and accuracy in train and in test (other than the the difference in data) ?
They both use AverageMeter().
```python
#IN EPOCH()
# indicates whether the model is in eval or train mode (some layers behave differently in train and eval)
model.eval() if optimizer is None else model.train()
# compute metrics
prec1, prec5 = accuracy(output, target, topk=(1, 5))
batch_time = time.time() - tic
tic = time.time()

# update
avg_loss.update(loss.item())
avg_top1_acc.update(prec1.item())
avg_top5_acc.update(prec5.item())
avg_batch_time.update(batch_time)
if optimizer:
	loss_plot.update(avg_loss.val)


# IN MAIN
# Train phase
top1_acc, avg_top5_acc, loss = epoch(train, model, criterion, optimizer, cuda)
# Test phase
top1_acc_test, top5_acc_test, loss_test = epoch(test, model, criterion, cuda=cuda)
```
In the epoch() function, if the parameter "optimizer" is None, then it calls model.eval():
```python
def eval(self):
	r"""Sets the module in evaluation mode."""
	return self.train(False)
```
Otherwise, it calls method model.train():
```python
def train(self, mode=True):
	r"""Sets the model in training mode."""
	self.training = mode
	for module in self.children():
		module.train(mode)
	return self
```
As is shown in the above codes, the model.train() sets the modules in the network in training mode. It tells our model that we are currently in the training phase so the model keeps some layers, like dropout, batch-normalization which behaves differently depends on the current phase, active. Whereas the model.eval() does the opposite. For instance, in training mode, BatchNorm updates a moving average on each new batch; whereas, for evaluation mode, these updates are frozen. Therefore, once the model.eval() has been called then, our model deactivate such layers so that the model outputs its inference as is expected.

Additionally, the error displayed during training is an average computed over different batches of data. This means it doesn't represent the final error at the end of an epoch, which is the case during testing. ( #TODO : revision)

### 15. Modify the code to use the CIFAR-10 dataset and implement the architecture requested above. (the class is datasets.CIFAR10 ). Be careful to make enough epochs so that the model has finished converging.
...

### 16.  What are the effects of the learning rate and of the batch-size ?
Convergence depends on the choice of η (eta, the learning rate): if it's too small, the model will take a long time to train, and if it's too large, the model may fail to converge. In practice, it's common to decrease η as training progresses.

Moreover, depending on the amount of data used to compute the gradient, we have:
- Batch gradient descent: The gradient is computed over the entire dataset for a single precise update, which can be slow.
- Stochastic gradient descent: It computes the gradient one example at a time with an update after each, making the training faster and suitable for online learning. However, the model may struggle to converge to the global minimum if the gradient step size doesn't decrease over iterations.
- Mini-batch gradient descent: The gradient is computed over a batch of examples, reducing the parameter update variance, resulting in more stable convergence.

Here we present extended experiments:

**Baseline**: batch_size=128, lr=0.1, epochs=20, cuda=True
![[Pasted image 20231027223453.png]]

**1. Changing Learning Rate:**
- **Too High:** A high learning rate can lead to rapid convergence, but it may overshoot the optimal weights and lead to divergence. This can result in the loss function increasing rather than decreasing. Training may become unstable, and the model may not converge to a good solution.
Experiment: epochs=20; lr=1, batch_size=128
![[Pasted image 20231027223729.png]]

- **Too Low:** A low learning rate can make the training process very slow. The model may get stuck in local minima or plateaus, and it may require a large number of epochs to converge. It's also sensitive to noise in the data.
Experiment: epochs=20; lr=0.001, batch_size=128
![[Pasted image 20231027223928.png]]
Experiment: epochs=20; lr=0.00001, batch_size=128
![[Pasted image 20231027224138.png]]


**2. Changing Batch Size:**
- **Small Batch Size:** Using a small batch size (e.g., 1, 32) can result in noisy updates. It can lead to faster convergence within an epoch, but the model's training can be less stable due to the randomness of each batch. It may also require more epochs to converge, and it may get stuck in local minima.
Experiment: epochs=20; lr=0.1, batch_size=32
![[Pasted image 20231027224604.png]]
Experiment: epochs=20; lr=0.1, batch_size=8
Time computation: 37seconds per each epoch (which is a lot more than the previous ones)
![[Pasted image 20231027230737.png]]

- **Large Batch Size:** A large batch size (e.g., 64, 128, 256) can provide more stable and accurate gradient estimates, resulting in smoother training curves. However, it may require more memory and can be computationally expensive. It might also converge to a wider minima rather than the optimal minima, which could affect generalization.
Experiment: epochs=20; lr=0.1, batch_size=256
![[Pasted image 20231027231300.png]]


### 17. What is the error at the start of the first epoch, in train and test ? How can you interpret this ?
In the first epoch we get:
[TRAIN Batch 000/469]	Time 0.104s (0.104s)	Loss 2.3017 (2.3017)	Prec@1   7.8 (  7.8)	Prec@5  50.8 ( 50.8)
![[Pasted image 20231021171542.png]]
This is because if you create a line plot with just a single value using `plt.plot([single_value])`, it should not display a point or a line. Instead, it should create an empty Cartesian graph without any visible data.
If you would like to see a single point plotted, you would have to modify the parameters of the plot function, for instance, like this:
```python
# Creating a point marker for the single value
plt.plot([0], [single_value], marker='o', markersize=5)
```

From the text output, we can see that there is still a loss and accuracy that is calculated, but that is because of the random initialization of the network's weights, since the model is not trained yet. This serves as a kind of baseline: during training, the model should reduce this error. If it doesn't, it indicates that it is struggling to learn.
### 18. Interpret the results. What’s wrong ? What is this phenomenon ?
Experiment: batch_size=128, lr=0.1, epochs=50, cuda=True
![[Pasted image 20231027132228.png]]
These plots show the accuracy percentage over the number of epochs (on the left) and the loss number over the number of epochs (on the right). Our goal is to minimize the loss as much as possible and to maximize the accuracy as much as possible. The different metrics are calculated on the train set (in blue) and on the test set (in orange). It's worth notice that the "test set" is not intended to be the set of images from the dataset used to make the _predictions_ after the training, but it is instead used for evaluating our training, therefore it is calculated during the training itself. More precisely, we use "test set", meaning "val test".

As we can see, the accuracy train and is increasing, while the loss train is decreasing and for us it's good. Instead, the loss test has a "U" shape, meaning that at a certain point it stops decreases and starts increase. This is a fenomenon called overfitting, where the model is becoming too specialized to the training data and performs poorly on unseen data. Also, the test accuracy tends to flatten at 70% instead of trying to achieve a better result. Thus, we should change our hyperparameters or the architecture of the model or improve the dataset or use other metrics in order to fix this issue.

This shows how the evaluation set (here called test set) is truly important: if we were evaluating ur model on the train set only, we would have thinked that the model performed well, but that it is not correct. 

---
# 3.1 Standardization of examples
### 19. Describe your experimental results.
Experiment: batch_size=128, lr=0.1, epochs=50, cuda=True
![[Pasted image 20231027134641.png]]
The train imrpoved a little bit than before, because the accuracy test is a little bit higher than 70% and the minimum of the loss test is lower than the 1.0 of before. But still, we see a fenomenon of overfitting and the test and trains curves are different from each others.

( #TODO: revision)

### 20. Why only calculate the average image on the training examples and normalize the validation examples with the same image ?
Preprocessing should not be learned on the test dataset, as it would bias the obtained performance results.

( #TODO :revision)

### 21. Bonus : There are other normalization schemes that can be more efficient like ZCA normalization. Try other methods, explain the differences and compare them to the one requested.
#TODO 


---
# 3.2 Increase in the number of training examples by data increase
### 22. Describe your experimental results and compare them to previous results.
Experiment: batch_size=128, lr=0.1, epochs=50, cuda=True
![[Pasted image 20231027182339.png]]
From the last train, we can see a huge improvement on the accuracy and on the loss: the test accuracy as increased and almost reaches the 80%, while the loss test doesn't stop decreasing so soon as before.

### 23. Does this horizontal symmetry approach seems usable on all types of images ? In what cases can it be or not be ?
The random horizontal symmetry takes an image as a input and flips it horizontally (usually in the code, with probability p). As en example:
![[Pasted image 20231027141209.png]]
Some images, such as the car shown above, keeps the same semantic even after the horizontal flip. Since the other classes are animal or vehicles, the semantic stys the same. What could be a downside, is that if we have and image that is perfectly horizzontally symmetrical, after the transformation we get the same input image, but that is a really an unfortunate case. Instead, if you have a dataset containing letters or numbers, certain type of them could change drastically their semantics after the flipping. For example in this set of the MNIST dataset:
![[Pasted image 20231027142024.png]]
The zeros and the ones (where the one is written with only a straight line) won't change their semantics, while the other numbers (5,4,9,2,3) would be unrecognizable.

### 24. What limits do you see in this type of data increase by transformation of the dataset ?
As descripted above, it is important that the semantics of the image stays the same, otherwise the network would learn some parameters that should not to. In other words, it is necessary that after a transformation the new image is useful for the training, to learn something for the ultimate goal.  ( #TODO : revision)

### 25. Bonus : Other data augmentation methods are possible. Find out which ones and test some.
#TODO 

--- 
# 3.3 Variants on the optimization algorithm

### 26. Describe your experimental results and compare them to previous results, including learning stability.
Experiment: batch_size=128, lr=0.1, epochs=50, cuda=True
![[Pasted image 20231027181232.png]]
The accuracy test has increased and the loss test has decreased a little bit more, but still the two curves differs a lot. We can also see that the loss test has been stabilized better. Remember that we chose a high number of epochs because we would like to express the training results in the most thruthful way as possible, but with an early stopping technique, one could simply decide to keep the weight that has the lowest loss or the highest accuracy.

### 27. Why does this method improve learning ?
As it is written here: https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler
A Learning rate schedule is a predefined framework that adjusts the learning rate between epochs or iterations as the training progresses. Two of the most common techniques for learning rate schedule are,
- Constant learning rate: as the name suggests, we initialize a learning rate and don’t change it during training; 
- Learning rate decay: we select an initial learning rate, then gradually reduce it in accordance with a scheduler.

It is achieved by adding the gradient calculated in the previous step, weighted by the gamma value = 0.95, to the new gradient before the update. This allows for an increased step when the direction remains the same and a decreased step when the direction changes, thereby accelerating the learning process.

With the learning rate scheduler we can move quickly in the beginning to approach a suitable solution rapidly. Once the model has converged, reducing the learning rate allows for fine-tuned improvements.

### 28. Bonus : Many other variants of SGD exist and many learning rate planning strategies exist. Which ones ? Test some of them.
#TODO

---
# 3.4 Regularization of the network by dropout

### 29. Describe your experimental results and compare them to previous results.
Experiment: batch_size=128, lr=0.1, epochs=50, cuda=True
![[Pasted image 20231027183751.png]]
After the modifications, we can see good improvements both in the accuracy and in the loss test functions. Expecially, we can see that the Dropout that we added lead to decrease significantly the overfitting fenomenon. 
Since a network with dropout requires more epochs to train, but for the same architecture, the epochs are computed faster because there are fewer parameters to update, we decided to increase the number of epochs:
Experiment: batch_size=128, lr=0.1, epochs=100, cuda=True
![[Pasted image 20231027210731.png]]
We can see that even with a number of epochs of 100 we don't have overfitting problems.

### 30. What is regularization in general ?
In general, regularization is a process that penalizes the complexity of a model, and it plays a vital role in mitigating the risk of overfitting.
Regularization comes in various forms, such as L1 (Lasso) and L2 (Ridge) regularization, dropout, weight decay, and early stopping, among others. Each of these techniques applies a different form of constraint on the model's parameters, effectively curbing its complexity and promoting better generalization.

### 31. Research and "discuss" possible interpretations of the effect of dropout on the behavior of a network using it
Dropout is a technique that involves "disabling" specific units in our neural network, thereby encouraging the learning of other units. During training, the gradient tends to increasingly favor certain pathways in the network, giving them more importance. By deactivating some units, the gradient is compelled to redistribute its learning across other parts of the network.
However, it's important to note that when dropout is applied, the network's expressiveness is reduced. To address this, it's often beneficial to increase the size of the neural network layers. This adjustment helps mitigate the risk of the model becoming overly reliant on specific units and enhances its ability to generalize, ultimately improving the model's overall performance and robustness.

### 32. What is the influence of the hyperparameter of this layer ?
The parameter 'p' corresponds to the probability of a neuron being deactivated through dropout. The choice of this probability is a crucial decision in implementing dropout effectively.

If 'p' is set too high, meaning that a substantial portion of neurons is deactivated during each training iteration, we run the risk of underfitting. Underfitting occurs when the model is too constrained and unable to capture the underlying patterns in the data, leading to poor performance.
Experiment:
#TODO
	
On the other hand, if 'p' is set too low, where only a small fraction of neurons are deactivated, we may not fully realize the benefits of dropout. In such cases, the model might still be prone to overfitting, as it doesn't experience enough regularization to prevent it from learning noise present in the training data.
Experiment:
#TODO

Choosing the right value for 'p' is a trade-off, and it often involves experimentation. The goal is to strike a balance between preventing overfitting and allowing the network to generalize effectively to new, unseen data.

### 33. What is the difference in behavior of the dropout layer between training and test ?
During the evaluation phase, dropout is disabled. Because you disable neurons _randomly_, your network will have different outputs every (sequences of) activation. This undermines consistency.
#TODO : revision

--- 
# 3.5 Use of batch normalization
### 34. Describe your experimental results and compare them to previous results.
![[Pasted image 20231027213348.png]]

The test accuracy improved again and also the loss has decreased again. Thanks to the dropout we don't see any overfitting effect.

We tried to use the increased number of epochs:
Experiment: batch_size=128, lr=0.1, epochs=100, cuda=True


## Conclusions
Al the expeeriments were run on a GPU NVIDIA A100 80GB.

The initial architecture was not good: the accuracy was too low (70%) and the loss presented some overfitting effects.
Our best model has obtained an accuracy evaluation of more than 80 % with a loss evaluation of 0.6 and without any overfitting effects. To further analise the model, there are other metrics that could be used such as precision, recall, f1, etc...
Another possibility is to train the model in a much larger dataset, such as ImageNet and choose a more complex architecture.