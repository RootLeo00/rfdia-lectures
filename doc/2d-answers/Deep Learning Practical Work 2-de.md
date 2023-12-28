## Introduction
#TODO 

---

**1 – Generative Adversarial Networks**
## 1. Interpret the equations (6) and (7). What would happen if we only used one of the two?
If we use only the loss of the equation (6) than the discriminator will not improve, therefore the generator will easily fool the discriminator and generate bad looking images.
If we only use equation (7) than the generator will not improve, therefore it will not pass the discriminator check.
## 2. Ideally, what should the generator G transform the distribution P (z) to?
Ideally the the generator should sample new images that can fool the discriminator check and that are similar to the ones in the train set but also original.
## 3. Remark that the equation (6) is not directly derived from the equation 5. This is justified by the authors to obtain more stable training and avoid the saturation of gradients. What should the “true” equation be here ?
Normally the loss function is this:
![](./images/Pasted%20image%2020231213162826.png)
so for the G the important part would be:
$$ min_G = E_{z\sim P(z)}log(1-D(G(z)))$$
The equation used by the authors is used in order to avoid the vanishing gradient problem:
during the optimization process if we calculate the gradient of this part of the loss it would look like this:
$$D[log(1-D(G(z)))] = D[\frac{1}{(1-\sigma)}\sigma] = \frac{\sigma(1-\sigma)}{(1-\sigma)} = \sigma$$
And at the beginning this would be closer to zero.

Instead if we rewrite this loss as:
$$max_G = log(D(G(z)))$$
then the derivation would be:
$$D[log(D(G(z)))] = \frac{1}{(\sigma)}\sigma(1-\sigma) = 1 -\sigma$$
which in the beginning would be going to 1

## 4. Comment on the training of of the GAN with the default settings (progress of the generations, the loss, stability, image diversity, etc.
#TODO 
We train our network with the given parameters, and we obtain images that are quite good and diverse. The loss curves of the generator and discriminator are not stable because the two networks are adversaries. When the generator's loss decreases, the discriminator's loss increases, and vice versa. This is because they are in a continual adversarial competition – when one network tries to reduce its loss, the other network responds by attempting to increase it, creating a dynamic and fluctuating relationship.

***Baseline**:*
*ngf=32 -> Channel size before the last layer in Generator*
*ndf=32 -> Channel size in Discriminator*
*weight_init="custom" -> Weight initialization type*
*loss_type=default -> Type of training loss for the generator*
*lr_d=0.0002 -> Learning rate for the discriminator*
*lr_g=0.0005 -> Learning rate for the generator*
*beta1=0.5 -> Beta1 for Adam optimizer*
*epochs=5 -> Number of training epochs*
*nz=100 -> Latent size*
*batch_size=12 -> Batch size*
*nchannels=1 -> Number of channels for inputs of Discriminator*
![300](./images/Pasted%20image%2020231228111114.png|300)![300](./images/Pasted%20image%2020231228112219.png|300)


## 5. Comment on the diverse experiences that you have performed with the suggestions above. In particular, comment on the stability on training, the losses, the diversity of generated images, etc.
#TODO 

**Modify ngf or ndf. In particular, reduce or increase one of the two significantly.**
Starting with ngf, if we increase it, we will not obtain significant differences between the baseline. We can see that the discriminator perform a little bit worse that the generator when increasing ndf and also the training seems more stable.
ngf=64
![300](./images/Pasted%20image%2020231228113811.png|300)![300](./images/Pasted%20image%2020231228113841.png|300)
ngf=128
![300](./images/Pasted%20image%2020231228113921.png|300)![300](./images/Pasted%20image%2020231228113936.png|300)
ngf=256
![300](./images/Pasted%20image%2020231228114012.png|300)![300](./images/Pasted%20image%2020231228114031.png|300)

While, increasing the ndf will produce different results: at ndf=256 the performances drop significantly. We can see that the generator loss perform poorer and poorer each time we increase ndf. One explanation could be that the discriminator becomes too powerful compared to the generator (i.e., it has a higher capacity). Moreover if the discriminator is too complex, the gradients during backpropagation may become very small (vanish), making it challenging for the generator to learn and update its parameters effectively.
ndf=64
![300](./images/Pasted%20image%2020231228114134.png|300)![300](./images/Pasted%20image%2020231228114145.png|300)
ndf=128
![300](./images/Pasted%20image%2020231228114224.png|300)![300](./images/Pasted%20image%2020231228114234.png|300)
ndf=256
![300](./images/Pasted%20image%2020231228114302.png|300)![300](./images/Pasted%20image%2020231228114321.png|300)

**Change the learning rate of one or both models**
Increasing the learning rate of one or both the models will produce really bad results and an unstable training:
lr_d=0.1
![300](./images/Pasted%20image%2020231228114943.png|300)![300](./images/Pasted%20image%2020231228115259.png|300)
lr_d=0.1, lr_g=0.1
![300](./images/Pasted%20image%2020231228115037.png|300)![300](./images/Pasted%20image%2020231228115222.png|300)
lr_g=0.1
![300](./images/Pasted%20image%2020231228115148.png|300)![300](./images/Pasted%20image%2020231228115206.png|300)

#TODO lr decreasing

**Learn for longer (ex : 30 epochs) even if it seems that the model already generates correct images**
After step 3000 we can see that the training gets more unstable: the two losses experience periods of diverging moment and then a spike. In our opinion, this could be the effect of an exploding gradient phenomenon.  Also the generator collapses to producing a limited set of samples, ignoring the diversity present in the training data. This is the "mode collapse problem" resulting in generated samples that lack variety.
epochs=30
![300](./images/Pasted%20image%2020231228115823.png|300)![300](./images/Pasted%20image%2020231228115835.png|300)
#TODO epochs 100

**Reduce or increase significantly nz (ex : nz = 10 ou 1000)**
Increasing the nz significantly will make the training more unstable: a very high-dimensional latent space might lead to overparameterization, where the model has more parameters than necessary. Overparameterization can make the training process slower and potentially lead to overfitting. Also a larger latent space requires more diverse and plentiful data to effectively capture the underlying data distribution. If the dataset is not sufficiently large or diverse, the model might struggle to learn meaningful representations.
nz=100
![300](./images/Pasted%20image%2020231228120800.png|300)![300](./images/Pasted%20image%2020231228121144.png|300)
If we decrease it, we obtain good results compared to the baseline. A lower-dimensional latent space simplifies the model, making it computationally more efficient and reducing the risk of overfitting. With fewer parameters, the model might generalize better to unseen data.
![300](./images/Pasted%20image%2020231228121211.png|300)![300](./images/Pasted%20image%2020231228121251.png|300)

#TODO point 2
#TODO point 7




---

**2 – BONUS : Conditional Generative Adversarial Networks**
## 6. Comment on your experiences with the conditional DCGAN.
#TODO

## 7. Could we remove the vector y from the input of the discriminator (so having cD(x) instead of cD(x, y)) ?
It is important that the discriminator and generator have access to the y label to generate a desired digit. If we remove the label y from the discriminator inputs, the model will not learn to generate under a condition. The generated numbers will be random like a normal GAN.

## 8. Was your training more or less successful than the unconditional case ? Why ?
#TODO

## 9. Test the code at the end. Each column corresponds to a unique noise vector z. What could z be interpreted as here ?
#TODO

