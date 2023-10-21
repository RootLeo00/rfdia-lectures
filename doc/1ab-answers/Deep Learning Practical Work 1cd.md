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

### 5. Show that we can analyze fully-connected layers as particular convolutions.

Mathematically, the operation of a fully-connected layer and a convolutional layer with a kernel size equal to the input size is the same. In both cases, each output unit computes a weighted sum of all the input values.

- In a fully-connected layer, if we consider the weight matrix W as the convolutional kernel, and the input as a 2D or 3D grid, then the output is given by O=W⋅I, where O is the output, W is the weight matrix, and I is the input.
    
- In a convolutional layer with a kernel size matching the input size, the convolution operation is represented as O=I∗K, where O is the output, I is the input, and K is the convolutional kernel. If the size of K is the same as the size of W, this is equivalent to the fully-connected layer operation.
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

[source](https://cs231n.github.io/convolutional-networks/#convert)
### 7. We call the receptive field of a neuron the set of pixels of the image on which the output of this neuron depends. What are the sizes of the receptive fields of the neurons of the first and second convolutional layers ? Can you imagine what happens to the deeper layers ? How to interpret it ?

![[Pasted image 20231013153617.png]]
[reference](https://cs231n.github.io/convolutional-networks/#conv)
- As you move deeper into the network, the receptive field size continues to increase. This is primarily due to the stacking of convolutional layers and pooling layers with larger windows or strides.
- Deeper layers have progressively larger receptive fields, allowing them to capture more global information and high-level features in the input image.
- This increase in receptive field size is a fundamental aspect of hierarchical feature learning in deep CNNs. Features learned in deeper layers tend to represent more complex and abstract patterns.
