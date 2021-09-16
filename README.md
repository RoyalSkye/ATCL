## AT with Complementary Labels

### Run
The following demo will show the results with the MNIST dataset.  After running the code, you should see a text file with the results saved in the same directory.  The results will have three columns: epoch number, training accuracy, and test accuracy.

```shell
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py 2>&1 &
```

#### Methods and models
In `main.py`, specify the `method` argument to choose one of the 5 methods available:

- `ga`: Gradient ascent version (Algorithm 1) in [1].
- `nn`: Non-negative risk estimator with the max operator in [1].
- `free`: Assumption-free risk estimator based on Theorem 1 in [1].
- `forward`: Forward correction method in [2].
- `pc`: Pairwise comparison with sigmoid loss in [3].

Specify the `model` argument:

- `linear`: Linear model
- `mlp`: Multi-layer perceptron with one hidden layer (500 units)
- `resnet`: ResNet-34

#### Results on CIFAR-10 (CL) - [Logs](https://drive.google.com/drive/folders/1EhzJDNdAbWm6yGQ8yev128leVsjXji3p?usp=sharing)

> Settings: For CIFAR-10, ResNet-34 was used with weight decay of 5e−4 and initial learning rate of 1e−2. For optimization, SGD was used with the momentum set to 0.9. Learning rate was halved every 30 epochs. We train the model for 300 epochs with batch_size = 256.
>
> **Currently, AT failed on `CIFAR10` (MNIST is ok), mainly due to the low nature acc of the model. If warmup for several epochs + decrease the num steps of PGD, AT can work normally.** 
>
> *Matrix in logs:*
>
> First column: `true label` of an image; Last column: original `complementary label` of an image; Others: the model prediction of generated adv examples as the pgd-100 progress, which can be viewed as extra (multiple) complementary labels.
>
> tensor([[2, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...])

|             CL              | nature_test_acc (Last / Best) |
| :-------------------------: | :---------------------------: |
| ~~unbiased_risk_estimator~~ |        ~~9.29 / 26.2~~        |
|      free (Ishida 19)       |         11.61 / 29.15         |
|       nn (Ishida 19)        |         23.78 / 35.27         |
|       ga (Ishida 19)        |         31.43 / 31.67         |
|      ~~modified_free~~      |       ~~11.68 / 24.00~~       |
|       ~~modified_nn~~       |       ~~22.50 / 34.73~~       |
|       ~~modified_ga~~       |       ~~26.91 / 27.42~~       |
|         ~~scl_exp~~         |       ~~10.00 / 10.00~~       |



|           **AT**           | nature_test_acc |
| :------------------------: | :-------------: |
|  **CIFAR10 - ResNet-34**   |                 |
|     adv_min_nn_max_nn      |        Failed        |
|     adv_min_nn_min_ce      |        F        |
|   adv_min_free_max_free    |        F        |
|      min_ure_max_ure       |        F        |
| warmup + min_free_max_free |       F+        |
|  warmup + min_free_min_ce  |       F+        |
|      **MNIST - MLP**       |                 |
|     min_free_max_free      |      77.94      |
| warmup + min_free_max_free |      87.86      |
|  warmup + min_free_min_ce  |      88.34      |




### Reference
1. T. Ishida, G. Niu, A. K. Menon, and M. Sugiyama.<br>**Complementary-label learning for arbitrary losses and models**.<br>In *ICML 2019*. [[paper]](https://arxiv.org/abs/1810.04327)
2. Yu, X., Liu, T., Gong, M., and Tao, D.<br>**Learning with biased complementary labels**.<br>In *ECCV 2018*. [[paper]](https://arxiv.org/abs/1711.09535)
3. T. Ishida, G. Niu, W. Hu, and M. Sugiyama.<br>**Learning from complementary labels**.<br>In *NeurIPS 2017*. [[paper]](https://arxiv.org/abs/1705.07541)

