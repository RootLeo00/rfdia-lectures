*Authors:*
*Caterina Leonelli: 21306668*
*Luisa Neubauer: 21228141*
## Domain Adaptation
#### Introduction
The information we use can come from different places. Let's take an example related to self-driving cars. There's data used for figuring out things like where the road is, where the cars are, and where people are. Imagine one set of pictures is from a video game called GTA-5, and another set is from real streets. Even though both sets show the same things like roads, cars, and people, the pictures look quite different. Also, getting data with labels (like saying what's in each picture) can be really hard and expensive, taking a lot of time and money. That's where domain adaptation comes in. It's about training a computer on one set of labeled pictures (from GTA-5 where the game tells us what's what) and making sure it works well on another set of pictures without labels. 
In this practical work, we trained the DANN network, composed by 3 blocks: the feature extractor (green), the classifier (blue) and the domain classifier (pink). The feature extractor and the domain classifier can be seen as a generator and a discriminator respectively. Therefore the training would look like an adversarial training, where the generator aims to fool the discriminator by outputting features of the source and target domain that are meshed well in the latent space. This is also possible by applying the gradient reversal layer: the generator "un-learns" features of the domains that are necessary to distinguish the domains but un-necessary for the final task.   
![[Pasted image 20231229142728.png]]
![[Pasted image 20231229142741.png]]

#### 1. If you keep the network with the three parts (green, blue, pink) but didn’t use the GRL, what would happen ?
The GRL is used to optimize the features (output of the green network) so that the domain classifier becomes unable to distinguish the target domain from the source domain. In the case of MNIST and MNISTM dataset, if the domain gradient is not reversed than the feature space that comes as the output of the green network would have two distinctive clusters, therefore the classification of the digits would perform bad.
#### 2. Why does the performance on the source dataset may degrade a bit ?
The performances degrade because because the green network will not effectively mitigate domain-specific variations in the input data. This could result in poor performance when the model is applied to data from a domain that is different from the one it was trained on.
#### 3. Discuss the influence of the value of the negative number used to reverse the gradient in the GRL.
We apply a factor that at the beginning is high so that the pink part loss is valued more strongly, then we decrease that value in order to make the green part learn better since that is our goal.

#### 4. Another common method in domain adaptation is pseudo-labeling. Investigate what it is and describe it in your own words.
Pseudo-labeling is a five-step approach to enhancing model performance:
1. Train a model using your labeled training data.
2. Use the trained model to predict labels for an unseen test dataset.
3. Add confidently predicted test observations to your training data. These predictions serve as pseudo-labels.
4. Build a new model using the combined dataset, which includes both the original training data and the newly augmented data from step 3.
5. Finally, use the newly trained model to make predictions on the test data

With these steps the model obtains additiional information provided by the target domain even if that information is not fully reliable due to the use of pseudo-labels. Iterating this process helps refine the model's understanding of the target domain and encourages the learning of domain-invariant features.
However, the success of pseudo-labeling depends on factors such as the similarity between the source and target domains. If the domains are too dissimilar, or if the pseudo-labels are noisy, the benefits of pseudo-labeling may diminish.

source: 
https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969
https://www.mdpi.com/2079-9292/12/15/3325