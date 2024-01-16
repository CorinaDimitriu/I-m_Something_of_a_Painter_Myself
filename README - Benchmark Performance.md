## Important components ##

* Kaggle notebook: https://www.kaggle.com/code/corinacode/gan-getting-started
* Competition score: 39.87871 (8th place)

***
The solution builds a Cycle GAN model (having a U-Net model as generator) with a suite of improvements regarding the loss functions and parameters: 
* from literature - U-Net based discriminator, Top-k training, "Better Cycles"
* own contribution - output-to-input technique (see the Report).

***
Major changes: the weights for the three losses comprised by the generator within the total generative loss remain the most influential component.
