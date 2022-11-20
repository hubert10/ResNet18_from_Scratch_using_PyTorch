## Training ResNet18 from Scratch using PyTorch

This repo trains compared the performance of two models
trained on the same datasets. We replicated the ResNet18 neural network model from scratch using PyTorch. Only creating a model is not enough. We need to verify whether it is working (able to train) properly or not.

For that reason, we will train it on a simple dataset. And to check that indeed it is doing its job, we will also train the Torchvision ResNet18 model on the same dataset. The technical details will follow in the next sections.

### The CIFAR10 Dataset
Anyone who has been in the field of deep learning for a while is not new to the famous CIFAR10 dataset.

The CIFAR10 dataset contains 60000 RGB images each of size 32×32 in dimension.

Out of the 60000 images, 50000 are for training and the rest 10000 for testing/validation.

All the images in the CIFAR10 dataset belong to one of the following 10 classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

CIFAR10 is a good dataset to test out any custom model. If it is able to achieve high accuracy on this dataset, then it is probably correct and will train on other datasets as well.

### ResNet18 from Scratch Training

In this subsection, we will train the ResNet18 that we built from scratch in the last tutorial.

All the code is ready, we just need to execute the train.py script with the --model argument from the project directory.

**python train.py --model scratch**

By the end of 20 epochs, we have a training accuracy of 98% and a validation accuracy of 73.24%. But looking at the graphs will give us more insights.

<img src="https://github.com/hubert10/ResNet18_from_Scratch_using_PyTorch/blob/main/outputs/resnet_scratch_accuracy.png"/> 

<img src="https://github.com/hubert10/ResNet18_from_Scratch_using_PyTorch/blob/main/outputs/resnet_scratch_loss.png"/> 


Although the training looks pretty good, we can see a lot of fluctuations in the validation accuracy and loss curves. The CIFAR10 dataset is not the easiest of the datasets. Moreover, we are training from scratch without any pretrained weights. But we will get to actually know whether our ResNet18 model is performing as it should only after training the Torchvision ResNet18 model.

### Torchvision ResNet18 Training

Now, let’s train the Torchvision ResNet18 model without using any pretrained weights.

**python train.py --model torchvision**

We can see a similar type of fluctuations in the validation curves here as well.


<img src="https://github.com/hubert10/ResNet18_from_Scratch_using_PyTorch/blob/main/outputs/resnet_torchvision_accuracy.png"/> 


<img src="https://github.com/hubert10/ResNet18_from_Scratch_using_PyTorch/blob/main/outputs/resnet_torchvision_loss.png"/> 

Most of these issues can be solved by using image augmentation and a learning rate scheduler.

But from the above experiments, we can conclude that our ResNet18 model built from scratch is working at least as well as the Torchvision one if not be.


### Summary and Conclusion

In this repo, we carried out the training of a ResNet18 model using PyTorch that we built from scratch. We used the CIFAR10 dataset for this. To compare the results, we also trained the Torchvision ResNet18 model on the same dataset. We found out that the custom ResNet18 model is working well
