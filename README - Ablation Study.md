## Important components ##

* No destined Kaggle notebook was used for this component, the results were obtained on the GPU of the personal laptop, except: taking the notebook from the benchmark performance component and training it on a new round of previously obtained outputs (and, therefore, exceeding five hours of training) taken as inputs improves the score with ~0.3: 39.55551

***
A second solution based on stable diffusion and the generative adversarial framework's principles was implemented as own contribution: 
* The main idea of this new architecture is to create a collaborative and adversarial ---  both at the same time --- environment mostly through the generators' architecture itself. Therefore, the purpose of the double U-Net model is to create a chain so that, through the power of following the evolution of the same distribution, the paintings and the images put together their style and structure. This results in quicker contours drawing and artificially enlarges a small dataset, such as the one consisting in Monet's paintings exclusively and employed as it is by the vanilla Stable Diffusion model whose improvement is under discussion. Starting from noise, both generators approach their goal (photos or paintings) through the intermediate generation of the opposite goal (paintings or photos), which is ensured to be made part of the same distribution. As the training advances, photo drawing as a target implicitly encourages paintings drawing until the middle of the chain, when the roles are switched and the paintings serve as a base for generating photos; the same happens within the second generator, where drawing paintings implicitly encourages drawing the randomly paired photos until the middle of the chain, when the created photos themselves start to encourage drawing paintings.
* The key pinciple of the backward propagation is to avoid forward passes through the sable diffusion paradigm, since there were no resources available to achieve that. Even better, this can translate into a potential solution for avoiding such forward passes if they are found to be needed in future approaches. The fake pieces are represented by the middle result of the generator having the opposite goal. In the case of the paintings, the fake pieces will be the middle output --- that is, the output of the first U-Net --- of the photos generator, whereas the fake photos will be the middle output of the paintings generator. The loss specific to generators might keep the indentity and cycle functions having this new acception of fake pieces, but the choice of the evaluated implementation was to discard them as proceeded in the discriminator's case as well. Eventually, the essential components of the generator's loss is the difference between the induced noise and the noise predicted by the network for the sampled timestep --- the classical noise of a Stable Diffusion model --- and the difference between the noisy images sampled (the ones fed into the  model) and the fake pieces extracted from the middle of the opposite network The reason this might work is because the two networks are trained with the same sampled noise over the photos/paintings and, at a specific timestep, the middle piece should resemble the piece fed into the opposite network as much as possible. It is important to note the symmetry of the double U-Net model, which facilitate the logic of the forward step and the backward process to be consistent with each other.

***
"Vanilla" stable diffusion was also experienced, but the results provided by the solution above were better.

***
Keeping the discriminator for the second solution was also experienced, but the generation process was way too slow on the GPU I have; however, it showed visually comparable results to the version without discriminator, but on 32 bits.

***
Experiments on the literature review corresponding to the first solution: U-Net based discriminator, Top-k training, "Better Cycles", along with the influence of various parameters:
* batch size
* identity loss ratio
* cycle loss ratio
* output-to-input technique
* data post-processing

***
Findings:
* This second solution performs better than vanilla stable diffusion;
* The identity loss ratio influences the nuances, while the cycle loss ratio has more to do with the structure/content/contours;
* There are images left (almost) unmodified by the first solution, which encourages progress on the track started with the second solution, as it generates images from scratch;
* The most influential changes regarding the first solution: the identity and cycle loss ratios within the generator's loss; the output-to-input technique;
* Operating on larger and more diverse datasets gives more power to stable diffusion based models.