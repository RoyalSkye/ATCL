### Learning with Complementary Labels

#### Run
```shell
# EXP
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset 'kuzushiji' --cl_num=1 --method 'exp' 2>&1 &
# EXP+AT
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset 'kuzushiji' --cl_num=1 --method 'exp' --at 2>&1 &
```

#### Results

> Setting: All experimentsare under uniform assumption. For MNIST, Fashion-MNIST, and Kuzushiji- MNIST,  a MLP model (d − 500 − 10) was trained for 300 epochs. Weight decay of 1e − 4 for weight parameters and learning rate of 5e − 5 for Adam was used. For CIFAR-10, DenseNet and ResNet- 34 were used with weight decay of 5e − 4 and initial learning rate of 1e − 2. For optimization, stochastic gradient descent was used with the momentum set to 0.9. Learning rate was halved every 30 epochs.

##### Exp 1: Single Complementary Label

|           | ***MNIST*** | ***Fashion*** | ***Kuzushiji*** |
| :-------: | :---------: | :-----------: | :-------------: |
|    EXP    | 93.32(0.03) |  83.68(0.23)  |   65.46(1.1)    |
|    LOG    | 93.37(0.08) |  83.73(0.2)   |   66.41(0.44)   |
|    MAE    |             |               |                 |
|    MSE    |             |               |                 |
|    GCE    |             |               |                 |
| Phuber-CE |             |               |                 |
|    CCE    |             |               |                 |
|  SCL_EXP  |             |               |                 |
|  SCL_NL   |             |               |                 |
|   L-UW    |             |               |                 |
|    L-W    |             |               |                 |
| EXP+AT20. | 97.04(0.04) |  84.28(0.09)  |   81.21(0.26)   |
| LOG+AT20. | 97.06(0.1)  |  84.84(0.14)  |   81.18(0.49)   |

##### Exp 2: Multiple Complementary Labels

|               |     s=1     |     s=2     |     s=3     |     s=4     |     s=5     |     s=6     |     s=7     |     s=8     |
| :-----------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
|  ***MNIST***  |             |             |             |             |             |             |             |             |
|      EXP      | 93.32(0.03) | 95.39(0.11) | 96.3(0.07)  | 96.91(0.07) | 97.26(0.11) | 97.55(0.1)  | 97.77(0.02) | 97.94(0.05) |
|      LOG      | 93.37(0.08) | 95.34(0.17) | 96.37(0.11) | 97.0(0.09)  | 97.45(0.07) | 97.78(0.08) | 97.98(0.04) | 98.12(0.02) |
|   EXP+AT20.   | 97.04(0.04) | 97.73(0.06) | 98.0(0.06)  | 98.1(0.06)  | 98.24(0.05) | 98.27(0.06) | 98.42(0.05) | 98.41(0.02) |
|   LOG+AT20.   | 97.06(0.1)  | 97.79(0.03) | 98.14(0.09) | 98.26(0.07) | 98.41(0.07) | 98.51(0.03) | 98.59(0.02) | 98.68(0.03) |
| ***Fashion*** |             |             |             |             |             |             |             |             |
|      EXP      | 83.68(0.23) | 85.36(0.13) | 86.16(0.1)  | 86.67(0.05) | 87.09(0.09) | 87.42(0.1)  | 87.75(0.12) | 87.98(0.08) |
|      LOG      | 83.73(0.2)  | 85.45(0.25) | 86.35(0.12) | 87.36(0.1)  | 87.78(0.04) | 88.42(0.03) | 88.66(0.1)  | 89.22(0.06) |
|   EXP+AT20.   | 84.28(0.09) | 84.95(0.07) | 85.29(0.2)  | 85.38(0.15) | 85.59(0.15) | 85.61(0.1)  | 85.75(0.08) | 85.93(0.06) |
|   LOG+AT20.   | 84.84(0.14) | 85.77(0.17) | 86.37(0.11) | 86.67(0.03) | 87.03(0.08) | 87.36(0.12) | 87.52(0.09) | 87.68(0.08) |
| **Kuzushiji** |             |             |             |             |             |             |             |             |
|      EXP      | 65.46(1.1)  | 70.61(0.22) | 72.13(0.56) | 78.95(2.57) | 80.8(2.62)  | 86.9(0.38)  | 86.42(2.64) | 89.57(0.1)  |
|      LOG      | 66.41(0.44) | 71.36(0.4)  | 76.95(2.06) | 82.04(2.96) | 86.15(0.48) | 88.12(0.42) | 89.35(0.11) | 90.28(0.09) |
|   EXP+AT20.   | 81.21(0.26) | 82.18(2.96) | 86.11(2.75) | 84.74(2.72) | 83.12(0.24) | 83.74(0.16) | 83.6(0.67)  | 83.55(0.34) |
|   LOG+AT20.   | 81.18(0.49) | 85.02(1.38) | 88.66(0.16) | 89.86(0.18) | 90.4(0.15)  | 91.38(0.13) | 91.65(0.05) | 92.2(0.07)  |

##### Exp3: Ablation Study: 

>  Single Complementary Label, num_steps=1, epsilon=step_size=$\theta$
>
> Note: The capacity of linear model is not enough for +AT.

<p align="center">  
  <img src="./imgs/ablation_kuzushiji.png" alt="ablation" width="400" /></br>
</p>

|            |      MNIST      |     Fashion     |    Kuzushiji    |                 |                 |  Kuzushiji-5CL  |
| :--------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|            |     EXP+AT      |     EXP+AT      |     EXP+AT      |     LOG+AT      | EXP+AT (linear) |     EXP+AT      |
|  Baseline  |   93.32(0.03)   |   83.68(0.23)   |   65.46(1.1)    |   66.41(0.44)   |   61.23(0.22)   |   80.8(2.62)    |
| **1/255**  |   93.48(0.15)   |   84.25(0.16)   |   66.42(1.04)   |   67.52(0.21)   |   61.3(0.19)    |   79.35(0.48)   |
|   2/255    |                 |                 |   67.52(1.0)    |                 |                 |                 |
| **4/255**  |   94.62(0.25)   |   85.17(0.03)   |   70.09(0.5)    |   71.11(0.42)   | **61.39(0.11)** |   83.44(2.45)   |
|   6/255    |                 |                 |   71.83(0.66)   |                 |                 |                 |
| **8/255**  |   95.67(0.13)   | **85.42(0.07)** |   72.84(0.73)   |   73.55(0.27)   |   60.72(0.07)   | **84.38(2.58)** |
|   12/255   |                 |                 |   74.36(0.41)   |                 |                 |                 |
| **16/255** |   96.8(0.03)    |   84.88(0.04)   |   76.96(2.93)   |   80.91(0.39)   |   54.84(0.2)    |   83.23(0.2)    |
|   20/255   |   97.04(0.04)   |   84.28(0.09)   | **81.21(0.26)** |                 |                 |                 |
| **24/255** | **97.25(0.04)** |   84.04(0.42)   |   80.91(0.79)   | **81.13(0.34)** |   48.14(0.67)   |   82.86(0.21)   |
|   28/255   |                 |                 |   80.21(1.0)    |                 |                 |                 |
|   32/255   |   96.01(0.05)   |   84.5(0.13)    |   79.47(0.42)   |   80.25(0.28)   |   14.88(3.01)   | **85.72(2.65)** |
|   36/255   |                 |                 |   78.10(0.5)    |                 |                 |                 |
|   40/255   |                 |                 |   76.32(0.13)   |                 |                 |                 |
|   44/255   |                 |                 |   74.70(1.75)   |                 |                 |                 |
|   48/255   |                 |                 |   62.58(7.92)   |                 |                 |                 |
|   64/255   |                 |   79.95(3.46)   |   64.38(4.47)   |                 |                 |                 |



|   Kuzushiji / 16/255   |                 |
| :--------------------: | :-------------: |
|          EXP           |   65.46(1.1)    |
|    EXP+AT (random)     |   65.17(1.11)   |
|    EXP+AT (max_EXP)    | **76.96(2.93)** |
|    EXP+AT (min_EXP)    |   62.55(0.61)   |
|    EXP+AT (max_LOG)    |   74.53(0.26)   |
|    EXP+AT (min_LOG)    |   63.99(0.84)   |
| EXP+AT (min/max_ce_cl) |        F        |

#### Reference

1. Y. T. Chou, G. Niu, H. T. Lin, and M. Sugiyama.<br>**Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels**.<br>In *ICML 2020*. [[paper]](https://arxiv.org/abs/2007.02235)
2. T. Ishida, G. Niu, A. K. Menon, and M. Sugiyama.<br>**Complementary-label learning for arbitrary losses and models**.<br>In *ICML 2019*. [[paper]](https://arxiv.org/abs/1810.04327)
3. Yu, X., Liu, T., Gong, M., and Tao, D.<br>**Learning with biased complementary labels**.<br>In *ECCV 2018*. [[paper]](https://arxiv.org/abs/1711.09535)
4. T. Ishida, G. Niu, W. Hu, and M. Sugiyama.<br>**Learning from complementary labels**.<br>In *NeurIPS 2017*. [[paper]](https://arxiv.org/abs/1705.07541)

