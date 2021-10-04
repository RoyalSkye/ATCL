## AT with Complementary Labels

### Run
The following demo will show the results with the MNIST dataset.  After running the code, you should see a text file with the results saved in the same directory.  The results will have three columns: epoch number, training accuracy, and test accuracy.

```shell
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --method 'scl_exp' --dataset 'mnist' 2>&1 &
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
|  (epsilon) progressive min_sclexp_max_sclexp  |  12.18 / 90.03   | 9.91 / 35.83  |  9.86 / 7.74  |
|        two_stage scl_exp (100 epochs)         |  88.43 / 88.43   | 61.67 / 61.67 | 43.63 / 43.63 |
|        two_stage scl_exp (300 epochs)         |  89.24 / 89.31   | 67.38 / 67.87 | 53.92 / 53.37 |
|          resnet18 two_stage scl_exp           |  99.41 / 99.40   | 97.78 / 97.91 | 96.76 / 96.87 |

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
  <img src="./ATCL_result/newpro_mlp_forward_test_acc.png" alt="newpro_mlp_free" width="250" />
<img src="./ATCL_result/newpro_mlp_sclexp_test_acc.png" alt="newpro_mlp_sclexp" width="250" /></br>
	Figure 2: Nature & PGD20 test acc during progressive (**epsilon**) adversarial training (MLP): </br>pro_nn, pro_forward and pro_sclexp, respectively.
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

> Two-stage Methods:

<p align="center">  
  <img src="./ATCL_result/two_stage_mlp_sclexp.png" alt="two_stage_mlp_sclexp_100" width="250" />
  <img src="./ATCL_result/two_stage_mlp_sclexp1.png" alt="two_stage_mlp_sclexp_300" width="250" />
<img src="./ATCL_result/two_stage_resnet18_sclexp.png" alt="two_stage_resnet18_sclexp_100" width="250" /></br>
	Figure 4: Nature & PGD20 test acc during adversarial training: </br>two_stage_sclexp(MLP_100), two_stage_sclexp(MLP_300) and two_stage_sclexp(resnet18_100), respectively.
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
|    WRN-32-10 progressive min_forward_max_forward     |  11.38 / 15.80   | 10.32 / 13.92 | 10.17 / 13.27 |
| Warmup WRN-32-10 progressive min_forward_max_forward |  41.21 / 41.07   | 10.64 / 13.27 | 11.67 / 12.99 |
|              resnet34 two-stage scl_exp              |  50.06 / 48.53   | 19.26 / 30.56 | 19.48 / 28.97 |

<p align="center">   
	<img src="./ATCL_result/cifar10_pro_resnet34_forward_test_acc.png" 	alt="pro_resnet34_forward" width="190" />
  <img src="./ATCL_result/cifar10_pro_wrn32_10_forward_test_acc.png" 	alt="pro_wrn32_10_forward" width="190" />
  <img src="./ATCL_result/cifar10_warmup_pro_wrn32_10_forward_test_acc.png" 	alt="warmup_pro_wrn32_10_forward" width="190" />
  <img src="./ATCL_result/cifar10_two_stage_resnet34_sclexp.png" alt="two_stage_resnet34_sclexp" width="190" /></br>
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
> 4. attempt: assign different weights to mcls, needed to modify loss func (weights / + CE(f(x'), tl) );
>
>    1. Weight -> exponential moving average
>
> 5. try with mcl (assume already know the true labels)
>
> 6. data augmentation: group by CL, then the cl of mixup(x1, x2) is still the same.

|                   ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯                   |       |       | wrong_cl | avg_correct_cl |
| :--------------------------------------------------: | :---: | :---: | :------: | :------------: |
|                      **MNIST**                       |       |       |          |                |
|      progressive sclexp **min_ce_cl** random_cl      |       |   ①   |  40.64%  |      6.76      |
|      progressive sclexp **min_ce_pl** random_cl      |       |   ②   |  1.50%   |      1.45      |
|      progressive sclexp **max_ce_cl** random_cl      |       |   ③   |  95.35%  |      7.63      |
|      progressive sclexp **max_ce_pl** random_cl      |       |   ④   |  5.88%   |      7.26      |
|  progressive **min_sclexp_max_sclexp_cl** random_cl  |       |   ⑤   |  28.46%  |      2.95      |
|    progressive **min_sclexp_max_ce_pl** random_cl    |   ✔️   |   ⑥   |  5.97%   |      6.93      |
|    progressive **min_sclexp_min_ce_cl** random_cl    |       |   ⑦   |  49.33%  |      4.81      |
|  progressive **min_sclexp_min_sclexp_pl** random_cl  |   ✔️   |   ⑧   |  5.99%   |      7.15      |
|                                                      |       |       |          |                |
|  **warmup** progressive sclexp max_ce_cl random_cl   |       |   ⑨   |  62.10%  |      3.89      |
| reduced_lr_50 progressive sclexp max_ce_pl random_cl |       |   ⑩   |  5.61%   |      7.00      |
|       ② + do not remove the first generated cl       |       |   ⑪   |  6.22%   |      7.94      |
|       ④ + do not remove the first generated cl       |       |   ⑫   |  6.49%   |      8.77      |
|       ⑥ + do not remove the first generated cl       |   ✔️   |   ⑬   |  6.10%   |      6.86      |
|       ⑧ + do not remove the first generated cl       |   ✔️   |   ⑭   |  6.18%   |      6.59      |
|                                                      |       |       |          |                |
|                ① + weighted_mcl_loss                 |       |  ①_1  |  48.38%  |      5.09      |
|                ② + weighted_mcl_loss                 |       |  ②_1  |  1.19%   |      1.30      |
|                ③ + weighted_mcl_loss                 |       |  ③_1  |  58.51%  |      5.15      |
|                ④ + weighted_mcl_loss                 |   ✔️   |  ④_1  |  5.55%   |      6.30      |
|                ⑤ + weighted_mcl_loss                 |       |  ⑤_1  |  36.73%  |      2.81      |
|                ⑥ + weighted_mcl_loss                 |       |  ⑥_1  |  5.79%   |      6.23      |
|                ⑦ + weighted_mcl_loss                 |       |  ⑦_1  |  40.03%  |      2.98      |
|                ⑧ + weighted_mcl_loss                 |       |  ⑧_1  |  5.80%   |      6.25      |
|                ⑪ + weighted_mcl_loss                 |       |  ⑪_1  |  5.50%   |      3.15      |
|                ⑫ + weighted_mcl_loss                 |       |  ⑫_1  |  5.92%   |      6.19      |
|                ⑬ + weighted_mcl_loss                 |       |  ⑬_1  |  6.09%   |      5.77      |
|                ⑭ + weighted_mcl_loss                 |       |  ⑭_1  |  6.09%   |      5.76      |
|       given mcls + min_weighted_max_sclexp_cl        | ATCL  |   ⑮   |  0.00%   |      9.00      |
|    given all labels + min_weighted_max_sclexp_cl     | ATCL0 |   ⑯   | 100.00%  |      9.00      |
|                                                      |       |       |          |                |
|                     **CIFAR10**                      |       |       |          |                |
|      progressive sclexp **max_ce_pl** random_cl      |       | ATCL8 |          |                |
|                                                      |       |       |          |                |
|              min_weighted_max_weighted               |       | ATCL2 |          |                |
|        min_weighted_max_weighted + min_ce_pl         |       | ATCL6 |          |                |
|        min_weighted_max_weighted + max_ce_pl         |       | ATCL7 |          |                |
|                                                      |       |       |          |                |
|                                                      |       |       |          |                |
|                                                      |       |       |          |                |

<p align="center">   
	<img src="./ATCL_result/1.png" 	alt="1" width="190" />
  <img src="./ATCL_result/2.png" 	alt="2" width="190" />
  <img src="./ATCL_result/3.png" 	alt="3" width="190" />
  <img src="./ATCL_result/4.png" alt="4" width="190" /></br>
	<img src="./ATCL_result/5.png" 	alt="5" width="190" />
  <img src="./ATCL_result/6.png" 	alt="6" width="190" />
  <img src="./ATCL_result/7.png" 	alt="7" width="190" />
  <img src="./ATCL_result/8.png" alt="8" width="190" /></br>
	<img src="./ATCL_result/11.png" 	alt="11" width="190" />
  <img src="./ATCL_result/12.png" 	alt="12" width="190" />
  <img src="./ATCL_result/13.png" 	alt="13" width="190" />
  <img src="./ATCL_result/14.png" alt="14" width="190" /></br>
	weighted_mcl_loss</br>
	<img src="./ATCL_result/1_1.png" 	alt="1_1" width="190" />
  <img src="./ATCL_result/2_1.png" 	alt="2_1" width="190" />
  <img src="./ATCL_result/3_1.png" 	alt="3_1" width="190" />
  <img src="./ATCL_result/4_1.png" alt="4_1" width="190" /></br>
	<img src="./ATCL_result/5_1.png" 	alt="5_1" width="190" />
  <img src="./ATCL_result/6_1.png" 	alt="6_1" width="190" />
  <img src="./ATCL_result/7_1.png" 	alt="7_1" width="190" />
  <img src="./ATCL_result/8_1.png" alt="8_1" width="190" /></br>
	<img src="./ATCL_result/11_1.png" 	alt="11_1" width="190" />
  <img src="./ATCL_result/12_1.png" alt="12_1" width="190" />
  <img src="./ATCL_result/13_1.png" alt="13_1" width="190" />
  <img src="./ATCL_result/14_1.png" alt="14_1" width="190" /></br>
	<img src="./ATCL_result/15.png" alt="15" width="300" />
  <img src="./ATCL_result/16.png" alt="16" width="300" /></br>
</p>




### Reference

1. Y. T. Chou, G. Niu, H. T. Lin, and M. Sugiyama.<br>**Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels**.<br>In *ICML 2020*. [[paper]](https://arxiv.org/abs/2007.02235)
2. T. Ishida, G. Niu, A. K. Menon, and M. Sugiyama.<br>**Complementary-label learning for arbitrary losses and models**.<br>In *ICML 2019*. [[paper]](https://arxiv.org/abs/1810.04327)
3. Yu, X., Liu, T., Gong, M., and Tao, D.<br>**Learning with biased complementary labels**.<br>In *ECCV 2018*. [[paper]](https://arxiv.org/abs/1711.09535)
4. T. Ishida, G. Niu, W. Hu, and M. Sugiyama.<br>**Learning from complementary labels**.<br>In *NeurIPS 2017*. [[paper]](https://arxiv.org/abs/1705.07541)

