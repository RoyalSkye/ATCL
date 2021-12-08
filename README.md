## Adversarial Training with Complementary Labels

### Run

```shell
# For MNIST/Fashion/KMNIST
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset 'kuzushiji' --method 'exp' --epochs=100 --adv_epochs=300 2>&1 & 
# For CIFAR10
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset 'cifar10' --method 'exp' --epochs=50 --adv_epochs=100 2>&1 & 
```

> Setting: All experiments are under single CL with uniform assumption. For MNIST, Fashion-MNIST, and Kuzushiji-MNIST, a MLP model (d−500−10) was trained for 100(CL)+300(Adv) epochs. For learning with complementary labels (CL), weight decay of 1e−4 for weight parameters and learning rate of 5e−5 for Adam was used. For adversarial training, only learning rate was changed to 1e-3, and is divided by 10 at 150th and 250th epoch. The batch size is 256. 
>
> For CIFAR-10, ResNet18 and WRN-32-10 were trained for 50(CL)+100(Adv) epochs. For learning with complementary labels (CL), weight decay of 5e−4 and initial learning rate of 1e−2 for SGD was used, with the momentum set to 0.9. Learning rate was halved every 30 epochs. For adversarial training, only the learning rate was changed to 0.1, and is divided by 10 at 50th and 75th epoch. The batch size is 128.

|              Last / Best              | Natural Test Acc |      PGD20      |       CW        |
| :-----------------------------------: | :--------------: | :-------------: | :-------------: |
|            ***Kuzushiji***            |                  |                 |                 |
|      two_stage baseline (64.23%)      |   39.78/39.72    |   31.03/31.23   |   28.11/28.39   |
|          two_stage sample pl          |   42.99/42.99    |   31.89/32.52   |   28.14/27.82   |
|    two_stage argmax_weight min_ce     |   41.52/41.51    |   31.80/32.34   |   28.56/28.33   |
|    two_stage argmax_weight max_ce     |   41.54/41.49    |   31.97/32.47   |   27.92/27.76   |
| two_stage argmax_weight min_ce_max_ce |   41.27/41.25    |   32.25/32.62   |   29.34/30.01   |
|  two_stage mixup max_pl with min_cl   |   46.27/46.17    |   34.79/35.29   |   30.88/30.81   |
|      two_stage mixup top-2 class      |   45.30/45.33    |   34.54/35.13   |   31.04/30.72   |
|       two_stage mixup 10 class        |   47.44/47.45    |   35.18/35.72   |   30.48/30.48   |
|       two_stage sample pl (exp)       |        F         |        F        |        F        |
|       two_stage sample pl (log)       |        F         |        F        |        F        |
|      two_stage sample pl (p**2)       |   42.86/42.79    |   33.38/33.66   |   29.40/29.53   |
|                                       |                  |                 |                 |
|             ***Fashion***             |                  |                 |                 |
|      two_stage baseline (83.75%)      |   63.76/63.66    |   52.17/52.91   |   47.13/47.67   |
|          two_stage sample pl          |   63.62/63.90    |   52.97/53.70   |   47.63/47.97   |
|  two_stage mixup max_pl with min_cl   |   67.87/67.82    | **55.34/56.09** | **51.08/50.54** |
|      two_stage mixup top-2 class      |   67.22/67.20    |   53.39/53.70   |   46.72/46.07   |
|       two_stage mixup 10 class        |                  |                 |                 |
|      two_stage sample pl (p**2)       |   64.65/64.81    |   53.31/53.78   |   47.94/47.62   |
|                                       |                  |                 |                 |
|             ***CIFAR10***             |                  |                 |                 |
|        two_stage baseline (X%)        |                  |                 |                 |
|  two_stage mixup max_pl with min_cl   |                  |                 |                 |



#### Reference

1. Gao, Y., & Zhang, M. L.<br>**Discriminative Complementary-Label Learning with Weighted Loss**<br>In *ICML 2021*. [[paper]](http://proceedings.mlr.press/v139/gao21d/gao21d.pdf)
2. Feng, L., Kaneko, T., Han, B., Niu, G., An, B., & Sugiyama, M.<br>**Learning with Multiple Complementary Labels**<br>In *ICML 2020*. [[paper]](https://arxiv.org/abs/1912.12927v3)
1. Chou, Y. T., Niu, G., Lin, H. T., & Sugiyama, M.<br>**Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels**<br>In *ICML 2020*. [[paper]](https://arxiv.org/abs/2007.02235)
2. Ishida, T., Niu, G., Menon, A., & Sugiyama, M.<br>**Complementary-label learning for arbitrary losses and models**<br>In *ICML 2019*. [[paper]](https://arxiv.org/abs/1810.04327)
3. Yu, X., Liu, T., Gong, M., & Tao, D.<br>**Learning with biased complementary labels**<br>In *ECCV 2018*. [[paper]](https://arxiv.org/abs/1711.09535)
4. Ishida, T., Niu, G., Hu, W., & Sugiyama, M.<br>**Learning from complementary labels**<br>In *NeurIPS 2017*. [[paper]](https://arxiv.org/abs/1705.07541)

