## AT with Complementary Labels

### Run
The following demo will show the results with the MNIST dataset.  After running the code, you should see a text file with the results saved in the same directory.  The results will have three columns: epoch number, training accuracy, and test accuracy.

```shell
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset 'mnist' --progressive 2>&1 &
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


#### Week 10/04/2021

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
>
> 7. Test new progressive: 
>
>    ATCL0 - 150 epochs; ATCL00 - 100 epochs; ATCL10 - no progressive 150 epochs; ATCL000 - 150(100+50) epochs
>
> 8. generated adv examples attack cl_model, only 70% data are wrongly predicted.

##### 1. Two-stage baseline

|             MNIST              | Natural Test Acc |     PGD20     |      CW       |
| :----------------------------: | :--------------: | :-----------: | :-----------: |
| two_stage scl_exp (100 epochs) |  88.43 / 88.43   | 61.67 / 61.67 | 43.63 / 43.63 |
| two_stage scl_exp (300 epochs) |  89.24 / 89.31   | 67.38 / 67.87 | 53.92 / 53.37 |
|   resnet18 two_stage scl_exp   |  99.41 / 99.40   | 97.78 / 97.91 | 96.76 / 96.87 |
|          **CIFAR10**           |                  |               |               |
|   resnet34 two-stage scl_exp   |  50.06 / 48.53   | 19.26 / 30.56 | 19.48 / 28.97 |

<p align="center">  
  <img src="./ATCL_result/two_stage_mlp_sclexp.png" alt="two_stage_mlp_sclexp_100" width="190" />
  <img src="./ATCL_result/two_stage_mlp_sclexp1.png" alt="two_stage_mlp_sclexp_300" width="190" />
	<img src="./ATCL_result/two_stage_resnet18_sclexp.png" alt="two_stage_resnet18_sclexp_100" width="190" />
	<img src="./ATCL_result/cifar10_two_stage_resnet34_sclexp.png" alt="two_stage_resnet34_sclexp" width="190" /></br>
</p>

##### 2. CL/PL -> MCLs

|                     **MNIST**                      |      | wrong_cl | avg_correct_cl |
| :------------------------------------------------: | :--: | :------: | :------------: |
|     progressive sclexp **min_ce_cl** random_cl     |  ①   |  40.64%  |      6.76      |
|     progressive sclexp **min_ce_pl** random_cl     |  ②   |  1.50%   |      1.45      |
|     progressive sclexp **max_ce_cl** random_cl     |  ③   |  95.35%  |      7.63      |
|     progressive sclexp **max_ce_pl** random_cl     |  ④   |  5.88%   |      7.26      |
| progressive **min_sclexp_max_sclexp_cl** random_cl |  ⑤   |  28.46%  |      2.95      |
|   progressive **min_sclexp_max_ce_pl** random_cl   |  ⑥   |  5.97%   |      6.93      |
|   progressive **min_sclexp_min_ce_cl** random_cl   |  ⑦   |  49.33%  |      4.81      |
| progressive **min_sclexp_min_sclexp_pl** random_cl |  ⑧   |  5.99%   |      7.15      |
|      ② + do not remove the first generated cl      |  ⑨   |  6.22%   |      7.94      |
|      ④ + do not remove the first generated cl      |  ⑩   |  6.49%   |      8.77      |
|      ⑥ + do not remove the first generated cl      |  ⑪   |  6.10%   |      6.86      |
|      ⑧ + do not remove the first generated cl      |  ⑫   |  6.18%   |      6.59      |
|               ① + weighted_mcl_loss                | ①_1  |  48.38%  |      5.09      |
|               ② + weighted_mcl_loss                | ②_1  |  1.19%   |      1.30      |
|               ③ + weighted_mcl_loss                | ③_1  |  58.51%  |      5.15      |
|               ④ + weighted_mcl_loss                | ④_1  |  5.55%   |      6.30      |
|               ⑤ + weighted_mcl_loss                | ⑤_1  |  36.73%  |      2.81      |
|               ⑥ + weighted_mcl_loss                | ⑥_1  |  5.79%   |      6.23      |
|               ⑦ + weighted_mcl_loss                | ⑦_1  |  40.03%  |      2.98      |
|               ⑧ + weighted_mcl_loss                | ⑧_1  |  5.80%   |      6.25      |
|               ⑨ + weighted_mcl_loss                | ⑨_1  |  5.50%   |      3.15      |
|               ⑩ + weighted_mcl_loss                | ⑩_1  |  5.92%   |      6.19      |
|               ⑪ + weighted_mcl_loss                | ⑪_1  |  6.09%   |      5.77      |
|               ⑫ + weighted_mcl_loss                | ⑫_1  |  6.09%   |      5.76      |

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
</p>

##### 3. Given mcls

|                     MNIST                     |      | wrong_cl | avg_correct_cl |
| :-------------------------------------------: | :--: | :------: | :------------: |
|    given mcls + min_weighted_max_sclexp_cl    |  ⑮   |  0.00%   |      9.00      |
| given all labels + min_weighted_max_sclexp_cl |  ⑯   | 100.00%  |      9.00      |
|        given 9 mcls + min_LOG_max_LOG         |  21  |  0.00%   |      9.00      |
|        given 9 mcls + min_EXP_max_EXP         |  22  |  0.00%   |      9.00      |
|        given 5 mcls + min_LOG_max_LOG         |  23  |  0.00%   |      5.00      |
|        given 5 mcls + min_EXP_max_EXP         |  24  |  0.00%   |      5.00      |
|                  **CIFAR10**                  |      |          |                |
|        given 9 mcls + min_LOG_max_LOG         |      |  0.00%   |      9.00      |

<p align="center">   
	<img src="./ATCL_result/17.png" 	alt="1" width="250" />
  <img src="./ATCL_result/18.png" 	alt="2" width="250" />
	<img src="./ATCL_result/27.png" 	alt="2" width="250" /></br>
	<img src="./ATCL_result/19.png" 	alt="1" width="190" />
  <img src="./ATCL_result/20.png" 	alt="2" width="190" />
	<img src="./ATCL_result/21.png" 	alt="1" width="190" />
  <img src="./ATCL_result/22.png" 	alt="2" width="190" />
</p>

##### 4. Attack cl pretrained model

> generated adv examples attack cl_model, only 70% data are wrongly predicted.

##### 5. min_EXP_min_ce_cl + no progressive + no random cl

| MNIST                                   |           |          | Natural         | PGD20           | CW              |
| --------------------------------------- | --------- | -------- | --------------- | --------------- | --------------- |
| two_stage scl_exp (100 epochs)          | /         | /        | 88.43/88.43     | 61.67/61.67     | 43.63/43.63     |
| two_stage scl_exp (300 epochs)          | /         | /        | 89.24/89.31     | 67.38/67.87     | 53.92/53.37     |
| progressive min_EXP_min_ce_cl random_cl | 6.24%     | 4.73     | 86.81/88.91     | 40.31/43.57     | 23.46/20.11     |
| min_EXP_min_ce_cl random_cl             | 5.98%     | 6.97     | 88.17/88.17     | 59.13/59.13     | 46.14/46.14     |
| min_EXP_min_ce_cl 100 epochs            | 5.80%     | 6.87     | 89.11/89.11     | 59.21/59.21     | 44.22/44.22     |
| **min_EXP_min_ce_cl 300 epochs (ours)** | **5.94%** | **6.88** | **91.95/91.95** | **70.04/70.04** | **57.18/57.18** |
| **CIFAR10**                             |           |          |                 |                 |                 |
| resnet34 two-stage scl_exp              | /         | /        | 50.06 / 48.53   | 19.26 / 30.56   | 19.48 / 28.97   |
| min_EXP_min_ce_cl                       | 45%+      | 7.36+    |                 |                 |                 |

<p align="center">   
	<img src="./ATCL_result/23.png" 	alt="1" width="190" />
  <img src="./ATCL_result/24.png" 	alt="1" width="190" />
  <img src="./ATCL_result/25.png" 	alt="1" width="190" />
  <img src="./ATCL_result/26.png" 	alt="1" width="190" />
</p>

##### 6. data augmentation

> Group by CL, then the cl of mixup(x1, x2) is still the same.



#### Week 10/11/2021

1. cl -> cl_model -> pl -> AT(min_EXP_min_ce_cl) -> mcls -> **mcl_model**  -> new_pl
2. if 100% of pl is tl, compare ce with EXP/LOG, given 9 mcls + min_LOG/EXP_max_LOG/EXP + no progressive
3. min_EXP_max_ce_cl / min_EXP_max_ce_pl
4. give 2/4/6/8 mcls, cl learning / min_EXP_min_ce_cl

##### 1

> **Iteratively: EXP (300 epochs) -> cl_model (93%+) -> pl -> min_EXP_min_ce_cl (100 epochs) -> robust model (60%+) with mcls -> EXP (300 epochs) -> cl_model (?) -> new_pl -> min_EXP_min_ce_cl (100 epochs) -> robust model (?) with mcls -> ...**
>
> autoregressive way: 1CL -> cl_model (EXP) -> pl (U-pl) -> cl_model (EXP)
>
> 6% noise + 6/7CL -> 96%+
>
> 1. 100_EXP_EXP
>
> 93.6 CL -> 89.12, 58.98, 45.57 -> 93.76 -> 88.93 58.95 44.53 -> 93.68 ->87.92 57.89 43.93 -> 93.69 -> 88.47 58.20 44.28 -> 93.76 -> 88.99 59.83 45.01 -> 93.71 -> 87.22 57.09 44.05 -> ... -> 88.84 60.19 46.70 -> 93.76 -> 89.33 60.72 46.80
>
> 2. 100_EXP_LOG
>
> 93.6 CL -> 89.12 58.98 45.57 -> 9.8 F (5.96%, 6.44)
>
> 3. 100_LOG_EXP
>
> 93.4 CL -> 89.04 58.27 44.30 -> 93.76 -> 88.56 58.14 43.92 -> 9.8 F
>
> 4. 100_EXP_MAE
>
> 93.6 CL -> 89.12 58.98 45.57 -> 93.64 -> 89.33 58.30 43.80 -> 93.74 -> 87.98 56.78 43.01 -> 93.54 -> ...
>
> 5. 300_EXP_EXP
>
> 93.6 CL -> 91.65 69.21 56.73 -> 93.68 -> 91.90 69.56 56.36 -> 93.7 -> 91.46 69.94 57.21 -> 93.71 -> ...
>
> 6. CL_EXP_EXP
>
> 93.34 -> 93.41 -> 93.4 -> 93.42 -> 93.42
>
> 7. CL_EXP_LOG
>
> 93.34 -> 93.4 -> 93.41 -> 93.35 -> 93.43
>
> 7. CL_EXP_MAE
>
> 93.34 -> 93.36 -> 93.39 -> 93.23 -> 93.43
>
> 8. CL_sclexp_EXP
>
> 93.43 -> 93.52

|       **MNIST**        |  wrong_cl  | avg_correct_cl |   Natural (100E/300E)   |       PGD20       |        CW         |
| :--------------------: | :--------: | :------------: | :---------------------: | :---------------: | :---------------: |
|  two_stage (scl_exp)   |     NA     |       NA       |       88.43/89.31       |    61.67/67.87    |    43.63/53.37    |
|   min_EXP_min_ce_cl    | 5.80-5.94% |      6.87      |       89.11/91.95       |    59.21/70.04    |    44.22/57.18    |
| Iteratively 100 epochs |   ATCL0    |    3679122     | ATCL1 (EXP_LOG Failed)  | 3944283 (LOG_EXP) | 4044481 (EXP_MAE) |
| Iteratively 300 epochs |   ATCL00   |    3679357     | ATCL11 (EXP_LOG Failed) |                   | 4044555 (EXP_MAE) |
|     CL EXP_EXP_EXP     |    ATCL    |                |                         |                   |                   |
|     CL EXP_LOG_EXP     |   ATCL2    |                |                         |                   |                   |
|     CL EXP_MAE_EXP     |    ATCL    |                |                         |                   |                   |

##### 2

| **MNIST**                       | wrong_cl   | avg_correct_cl | Natural (100E/300E) | PGD20           | CW           |
| ------------------------------- | ---------- | -------------- | ------------------- | --------------- | ------------ |
| two_stage (scl_exp)             | NA         | NA             | 88.43/89.31         | 61.67/67.87     | 43.63/53.37  |
| min_EXP_min_ce_cl               | 5.80-5.94% | 6.87           | 89.11/91.95         | 59.21/70.04     | 44.22/57.18  |
| **min_LOG_min_ce_cl**           | 3.60%      | 5.49           | F - 9.80            | F - 9.80        | NA           |
| two_stage + min_EXP_max_EXP     | 6.57%      | 8.934          | 64.39/64.79         | 52.62/58.17     | NA/53.89     |
| **two_stage + min_LOG_max_LOG** | **6.57%**  | **8.93**       | **88.46/89.06**     | **61.96/68.33** | **NA/55.43** |
| **min_ce_max_ce**               | **0.00%**  | **NA**         | **89.78/90.84**     | **60.80/68.69** | **NA/52.42** |
| min_LOG_max_LOG                 | 0.00%      | 6.00           | F                   | F               | F            |
| min_LOG_max_LOG                 | 0.00%      | 7.00           | F                   | F               | F            |
| min_LOG_max_LOG                 | 0.00%      | 9.00           | 89.57/92.01         | 60.44/68.51     | NA/51.37     |
| min_EXP_max_EXP                 | 0.00%      | 6.00           | F                   | F               | F            |
| min_EXP_max_EXP                 | 0.00%      | 7.00           | F                   | F               | F            |
| min_EXP_max_EXP                 | 0.00%      | 9.00           | 64.80/65.03         | 53.60/58.27     | NA/54.20     |

##### 3

| **MNIST**          | wrong_cl   | avg_correct_cl | Natural (100E/300E) | PGD20       | CW          |
| ------------------ | ---------- | -------------- | ------------------- | ----------- | ----------- |
| two_stage(scl_exp) | 6.58%      | NA             | 88.43/89.31         | 61.67/67.87 | 43.63/53.37 |
| min_EXP_min_ce_cl  | 5.80-5.94% | 6.87           | 89.11/91.95         | 59.21/70.04 | 44.22/57.18 |
| min_EXP_max_ce_cl  | 5.37%      | 3.89           | 89.95               | 0.00        | NA          |
| min_EXP_max_ce_pl  | 6.39%      | 8.59           | 77.74/78.26         | 55.47/63.67 | NA/57.76    |

##### 4

| **MNIST** | wrong_cl | avg_correct_cl | Train acc (%) | Test acc (%) |
| --------- | -------- | -------------- | ------------- | ------------ |
| 1 (EXP)   | 0.00%    | 1.00           | 93.07         | 93.13        |
| 2         | 0.00%    | 2.00           | 95.60         | 95.47        |
| 3         | 0.00%    | 3.00           | 96.76         | 96.42        |
| 4         | 0.00%    | 4.00           | 97.49         | 96.95        |
| 5         | 0.00%    | 5.00           | 98.11         | 97.39        |
| 6         | 0.00%    | 6.00           | 98.40         | 97.45        |
| 7         | 0.00%    | 7.00           | 98.79         | 97.83        |
| 8         | 0.00%    | 8.00           | 99.12         | 97.97        |
| 9         | 0.00%    | 9.00           | 99.36         | 98.17        |
|           |          |                |               |              |
| TL+1 CL   | 100.00%  | 1.00           | 0.010         | 0.06         |
| TL+2 CL   | 100.00%  | 2.00           | 0.033         | 0.2          |
| TL+3 CL   | 100.00%  | 3.00           | 0.026         | 0.16         |
| TL+4 CL   | 100.00%  | 4.00           | 0.048         | 0.15         |
| TL+5 CL   | 100.00%  | 5.00           | 0.073         | 0.12         |
| TL+6 CL   | 100.00%  | 6.00           | 0.102         | 0.12         |
| TL+7 CL   | 100.00%  | 7.00           | 0.155         | 0.29         |
| TL+8 CL   | 100.00%  | 8.00           | 0.370         | 0.35         |
| TL+9 CL   | 100.00%  | 9.00           | 12.53         | 12.47        |

| **①②: Based on pretrained (scl_exp) cl_model with 93.88% test acc** | wrong_cl   | avg_correct_cl | Natural (100E/300E) | PGD20           | CW           |
| ------------------------------------------------------------ | ---------- | -------------- | ------------------- | --------------- | ------------ |
| two_stage (scl_exp)                                          | NA         | NA             | 88.43/89.31         | 61.67/67.87     | 43.63/53.37  |
| min_EXP_min_ce_cl                                            | 5.80-5.94% | 6.87           | 89.11/91.95         | 59.21/70.04     | 44.22/57.18  |
| 1CL + min_EXP_min_ce_cl                                      | 0.00%      | 1.00           | F - 11.35           | 11.35           | 11.35        |
| 2CL(1+1) + min_EXP_min_ce_cl                                 | 0.00%      | 2.00           | F                   | F               | F            |
| 3CL + min_EXP_min_ce_cl                                      | 0.00%      | 3.00           | 16.05/16.17         | 15.24/15.55     | NA/15.19     |
| 4CL + min_EXP_min_ce_cl                                      | 0.00%      | 4.00           | 50.13/81.47         | 36.01/59.08     | NA/49.40     |
| 5CL + min_EXP_min_ce_cl                                      | 0.00%      | 5.00           | 83.26/88.70         | 55.55/68.28     | NA/58.19     |
| 6CL + min_EXP_min_ce_cl                                      | 0.00%      | 6.00           | **87.03/90.42**     | **59.12/69.39** | **NA/58.20** |
| 7CL + min_EXP_min_ce_cl                                      | 0.00%      | 7.00           | 84.14/89.64         | 55.43/66.52     | NA/54.57     |
| 8CL + min_EXP_min_ce_cl                                      | 0.00%      | 8.00           | 81.59/84.50         | 53.55/63.31     | NA/50.07     |
| 9CL + min_EXP_min_ce_cl                                      | 0.00%      | 9.00           | 79.54/84.26         | 48.86/60.28     | NA/44.91     |

**Current problem:**

1. two_stage + min_LOG_max_LOG > two stage min_ce_max_ce
2. 93 -> 96



### Reference

1. Y. T. Chou, G. Niu, H. T. Lin, and M. Sugiyama.<br>**Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels**.<br>In *ICML 2020*. [[paper]](https://arxiv.org/abs/2007.02235)
2. T. Ishida, G. Niu, A. K. Menon, and M. Sugiyama.<br>**Complementary-label learning for arbitrary losses and models**.<br>In *ICML 2019*. [[paper]](https://arxiv.org/abs/1810.04327)
3. Yu, X., Liu, T., Gong, M., and Tao, D.<br>**Learning with biased complementary labels**.<br>In *ECCV 2018*. [[paper]](https://arxiv.org/abs/1711.09535)
4. T. Ishida, G. Niu, W. Hu, and M. Sugiyama.<br>**Learning from complementary labels**.<br>In *NeurIPS 2017*. [[paper]](https://arxiv.org/abs/1705.07541)

