## AT with Complementary Labels

### Run
The following demo will show the results with the MNIST dataset.  After running the code, you should see a text file with the results saved in the same directory.  The results will have three columns: epoch number, training accuracy, and test accuracy.

```shell
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --method 'scl_exp' --dataset 'mnist' 2>&1 &
```

#### Methods and models
In `main.py`, specify the `method` argument to choose one of the 7 methods available:

- `ga`: Gradient ascent version (Algorithm 1) in [2].
- `nn`: Non-negative risk estimator with the max operator in [2].
- `free`: Assumption-free risk estimator based on Theorem 1 in [2].
- `forward`: Forward correction method in [3].
- `pc`: Pairwise comparison with sigmoid loss in [4].
- `scl_exp`: Exponential loss in [1].
- `scl_nl`: Negative learning loss in [1].

Specify the `model` argument:

* `linear`, `mlp` (0.4M), `cnn` (4.43 M), `resnet18` (11.17 M), `resnet34` (21.28 M), `densenet` (0.07 M), `wrn` (46.16M)

#### Results - [Logs](https://drive.google.com/drive/folders/1EhzJDNdAbWm6yGQ8yev128leVsjXji3p?usp=sharing)

> Settings: 
>
> For `CIFAR-10`, ResNet-34 was used with weight decay of 5e−4 and initial learning rate of 1e−2. For optimization, SGD was used with the momentum set to 0.9. Learning rate was halved every 30 epochs. 
>
> For `MNIST`, MLP was used with weight decay of 1e-4 and fixed learning rate of 5e-5 for Adam optimizer.
>
> We train the model for 300 epochs with batch_size = 256.

|      CL Loss      | MNIST - nature_test_acc (Last / Best) | CIFAR10 - nature_test_acc |
| :---------------: | :-----------------------------------: | :-----------------------: |
| free (Ishida 19)  |             77.94 / 85.81             |       11.61 / 29.15       |
|  nn (Ishida 19)   |             89.43 / 91.05             |       23.78 / 35.27       |
|  ga (Ishida 19)   |             93.14 / 93.34             |       31.43 / 31.67       |
|  pc (Ishida 17)   |             84.75 / 85.52             |       12.71 / 25.52       |
|  forward (Yu 18)  |             93.86 / 93.89             |       47.57 / 47.69       |
| scl_exp (Chou 20) |             93.86 / 93.88             |       49.02 / 49.38       |
| scl_nl (Chou 20)  |             93.89 / 93.90             |       46.79 / 47.09       |

##### AT - MNIST

> We avdersarially train a MLP/CNN/Resnet18 for 100 epochs. 
>
> epsilon = 0.3, num_steps = 40, step_size = 0.01
>
> lr = 5e-5, weight_decay = 1e-4, optimizer = Adam
>

|                   Baseline                    | Natural Test Acc |     PGD20     |      CW       |
| :-------------------------------------------: | :--------------: | :-----------: | :-----------: |
|               min_free_max_free               |        F         |       F       |       F       |
|                 min_nn_max_nn                 |        F         |       F       |       F       |
|                 min_ga_max_ga                 |        F         |       F       |       F       |
|                 min_pc_max_pc                 |        F         |       F       |       F       |
|    (Resnet18/MLP) min_forward_max_forward     |        F         |       F       |       F       |
|    Resnet18 warmup min_forward_max_forward    |        F         |       F       |       F       |
|             min_sclexp_max_sclexp             |        F         |       F       |       F       |
|              min_sclnl_max_sclnl              |        F         |       F       |       F       |
|         progressive min_free_max_free         |  13.85 / 54.06   | 10.93 / 28.71 | 9.41 / 17.55  |
|           progressive min_nn_max_nn           |   9.58 / 63.94   | 9.56 / 29.78  | 9.56 / 13.92  |
|           progressive min_ga_max_ga           |   9.58 / 67.03   | 9.55 / 28.81  | 9.56 / 10.40  |
|           progressive min_pc_max_pc           |  11.35 / 29.20   | 11.35 / 21.21 | 11.34 / 16.01 |
|      progressive min_forward_max_forward      |  10.11 / 87.42   | 10.10 / 36.18 | 10.09 / 7.55  |
|       progressive min_sclexp_max_sclexp       |  10.23 / 87.17   | 10.10 / 36.59 | 10.09 / 8.99  |
|        progressive min_sclnl_max_sclnl        |  10.40 / 87.24   | 10.10 / 35.84 | 10.09 / 7.56  |
|    resnet18 progressive min_free_max_free     |  25.39 / 95.71   | 5.79 / 82.96  | 1.47 / 72.75  |
|      resnet18 progressive min_nn_max_nn       |  98.20 / 98.20   | 95.78 / 96.11 | 94.65 / 95.15 |
|      resnet18 progressive min_ga_max_ga       |  98.35 / 98.44   | 96.25 / 96.57 | 94.93 / 95.28 |
|      resnet18 progressive min_pc_max_pc       |  52.15 / 96.27   | 14.34 / 83.93 | 11.61 / 75.86 |
| resnet18 progressive min_forward_max_forward  |  99.18 / 99.17   | 96.49 / 96.73 | 94.86 / 95.11 |
|  resnet18 progressive min_sclexp_max_sclexp   |  99.18 / 99.18   | 96.81 / 96.81 | 95.39 / 95.59 |
|   resnet18 progressive min_sclnl_max_sclnl    |  99.22 / 99.23   | 96.63 / 96.82 | 94.88 / 95.18 |
|    cnn progressive min_forward_max_forward    |  99.04 / 99.17   | 96.66 / 96.91 | 94.67 / 95.41 |
|      (epsilon) progressive min_nn_max_nn      |  29.39 / 56.18   | 15.25 / 27.80 | 13.85 / 10.41 |
| (epsilon) progressive min_forward_max_forward |  12.45 / 90.51   | 9.88 / 34.94  |  9.86 / 7.16  |
|        two_stage scl_exp (100 epochs)         |  88.43 / 88.43   | 61.67 / 61.67 | 43.63 / 43.63 |
|        two_stage scl_exp (300 epochs)         |  89.24 / 89.31   | 67.38 / 67.87 | 53.92 / 53.37 |
|          resnet18 two_stage scl_exp           |      ATCL1       |               |               |

> Nature & PGD20 test acc during `progressive` adversarial training (MLP):

<p align="center">   
  <img src="./ATCL_result/pro_mlp_free_test_acc.png" alt="pro_mlp_free" width="190" />
	<img src="./ATCL_result/pro_mlp_nn_test_acc.png" alt="pro_mlp_nn" width="190" />
  <img src="./ATCL_result/pro_mlp_ga_test_acc.png" alt="pro_mlp_ga" width="190" />
	<img src="./ATCL_result/pro_mlp_pc_test_acc.png" alt="pro_mlp_pc" width="190" /></br>
	<img src="./ATCL_result/pro_mlp_sclexp_test_acc.png" alt="pro_mlp_sclexp" width="190" />
	<img src="./ATCL_result/pro_mlp_sclnl_test_acc.png" alt="pro_mlp_sclnl" width="190" />
	<img src="./ATCL_result/pro_mlp_forward_test_acc.png" alt="pro_mlp_forward" width="190" /></br>
	Figure 1: Nature & PGD20 test acc during progressive adversarial training (MLP): </br>free, nn, ga, pc, </br>scl-exp, scl-nl and forward, respectively.
</p>
<p align="center">  
  <img src="./ATCL_result/newpro_mlp_nn_test_acc.png" alt="newpro_mlp_nn" width="250" />
  <img src="./ATCL_result/newpro_mlp_forward_test_acc.png" alt="newpro_mlp_free" width="250" /></br>
	Figure 2: Nature & PGD20 test acc during progressive (epsilon) adversarial training (MLP): </br>pro_nn and pro_forward, respectively.
</p>


> Nature & PGD20 test acc during `progressive` adversarial training (ResNet-18):

<p align="center">  
  <img src="./ATCL_result/pro_resnet18_free_test_acc.png" alt="pro_resnet18_free" width="190" />
  <img src="./ATCL_result/pro_resnet18_nn_test_acc.png" alt="pro_resnet18_nn" width="190" />
  <img src="./ATCL_result/pro_resnet18_ga_test_acc.png" alt="pro_resnet18_ga" width="190" />
  <img src="./ATCL_result/pro_resnet18_pc_test_acc.png" alt="pro_resnet18_pc" width="190" /></br>
	<img src="./ATCL_result/pro_resnet18_sclexp_test_acc.png" alt="pro_resnet18_sclexp" width="190" />
	<img src="./ATCL_result/pro_resnet18_sclnl_test_acc.png" alt="pro_resnet18_sclnl" width="190" />
  <img src="./ATCL_result/pro_resnet18_forward_test_acc.png" alt="pro_resnet18_forward" width="190" /></br>
	Figure 3: Nature & PGD20 test acc during progressive adversarial training (Resnet-18): </br>free, nn, ga, pc, </br>scl-exp, scl-nl and forward, respectively.
</p>

> Two-stage Methods (MLP):

<p align="center">  
  <img src="./ATCL_result/two_stage_mlp_sclexp.png" alt="two_stage_mlp_sclexp_100" width="250" />
  <img src="./ATCL_result/two_stage_mlp_sclexp1.png" alt="two_stage_mlp_sclexp_300" width="250" /></br>
	Figure 4: Nature & PGD20 test acc during adversarial training (MLP): </br>two_stage_sclexp(100) and two_stage_sclexp(300), respectively.
</p>

##### AT - CIFAR10

>We avdersarially train a ResNet34 for 100 epochs. 
>
>epsilon = 0.031, num_steps = 10, step_size = 0.007
>
>lr = 1e-2, weight_decay = 5e-4, momentum = 0.9, optimizer = SGD

|                                                      | Natural Test Acc |     PGD20     |      CW       |
| :--------------------------------------------------: | :--------------: | :-----------: | :-----------: |
|     resnet34 progressive min_forward_max_forward     |  31.49 / 27.49   | 13.50 / 19.07 | 13.39 / 17.81 |
|    resnet101 progressive min_forward_max_forward     |        F         |       F       |       F       |
|    WRN-32-10 progressive min_forward_max_forward     |      ATCL3       |               |               |
| Warmup WRN-32-10 progressive min_forward_max_forward |  41.21 / 41.07   | 10.64 / 13.27 | 11.67 / 12.99 |
|              resnet34 two-stage scl_exp              |      ATCL2       |               |               |

<p align="center">   
	<img src="./ATCL_result/cifar10_pro_resnet34_forward_test_acc.png" alt="pro_resnet34_forward" width="250" /></br>
	Figure 5: Nature & PGD20 test acc during progressive adversarial training: resnet34-forward.
</p>


#### TODO

> dYAfs2Dm
>
> 1. two-stage - baseline (generate pseudo-label as true label using CL, then do AT directly)
>
> 2. two-stage - observe the process of generate_cl_steps / how to make use of the info of CL & pseudo-label
>
>    what if min -> max
>
>    what if use cl_model instead of model to make prediction, only min-max formultaion
>
> 3. modify progressive - mlp
>
> attempt: assign different weights to mcls, needed to modify loss func (weights / + CE(f(x'), tl) )

|                                                   |          | wrong_cl | avg_correct_cl |
| :-----------------------------------------------: | :------: | :------: | :------------: |
|    progressive sclexp min_ce_cl **random_cl**     | ATCL6ing |          |                |
|           progressive sclexp min_ce_pl            |  ATCL6   |  2.14%   |       /        |
|    progressive sclexp min_ce_pl **random_cl**     | ATCL7ing |  1.37%   |      1.36      |
|     progressive sclexp **CL_model min_ce_cl**     |  ATCL12  |  5.23%   |      1.59      |
|                   3 + random_cl                   |  ATCL8   |  5.42%   |      1.57      |
|              3 + progressive_epsilon              |  ATCL9   |  5.33%   |      1.45      |
| 3 -> x_adv generated by model instead of CL_model |  ATCL10  |  10.76%  |      3.05      |
|                ATCL10 + random_cl                 |  ATCL11  |  11.76%  |      2.90      |
|           progressive sclexp max_ce_pl            |          |          |                |

>3. [3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
>
>   [8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
>
>

### Reference

1. Y. T. Chou, G. Niu, H. T. Lin, and M. Sugiyama.<br>**Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels**.<br>In *ICML 2020*. [[paper]](https://arxiv.org/abs/2007.02235)
2. T. Ishida, G. Niu, A. K. Menon, and M. Sugiyama.<br>**Complementary-label learning for arbitrary losses and models**.<br>In *ICML 2019*. [[paper]](https://arxiv.org/abs/1810.04327)
3. Yu, X., Liu, T., Gong, M., and Tao, D.<br>**Learning with biased complementary labels**.<br>In *ECCV 2018*. [[paper]](https://arxiv.org/abs/1711.09535)
4. T. Ishida, G. Niu, W. Hu, and M. Sugiyama.<br>**Learning from complementary labels**.<br>In *NeurIPS 2017*. [[paper]](https://arxiv.org/abs/1705.07541)

