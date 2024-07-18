# Toy models

**small4M** and **small4S** are two toy models that can be used to get a grasp of how the different algorithms work.  
The `dexom_python/toy_models.py` module contains the code for producing these models, as well as the code for producing a dagNet model where the metabolites of successive layers are all interconnected.

Below is a diagram of the small4M model, together with the reaction weights provided for the example.  
With the given weights, the model has four equally optimal flux distributions (one of which is the absence of flux in all reactions).

![small4M](small4M.png)

Below is a diagram of the small4S model, together with the reaction weights provided for the example.  
With the given weights, the model has four equally optimal flux distributions (one of which is the presence of flux in all reactions).

![small4S](small4S.png)

Below is a diagram of the r13m10 model, together with the reaction weights provided for the example.
This is the model that is used in the test_dexom_python.py test suite.
With the given weights, the model has three equally optimal flux distributions.

![r13m10](r13m10.png)
