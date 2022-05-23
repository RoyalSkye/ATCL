## Adversarial Training with Complementary Labels

### Dependencies

* Python 3.8
* Scipy
* [PyTorch 1.11.0](https://pytorch.org)
* [AutoAttack](https://github.com/fra31/auto-attack)

### How to Run

#### Baseline

```shell
# Two-stage
python main.py --dataset 'kuzushiji' --model 'cnn' --method 'log' --framework 'two_stage' --cl_epochs 50 --adv_epochs 50 --cl_lr 0.001 --at_lr 0.01 --seed 1
# LOG
python main.py --dataset 'kuzushiji' --model 'cnn' --method 'log' --framework 'one_stage' --adv_epochs 100 --at_lr 0.01 --scheduler 'none' --seed 1
```

#### Ours

```shell
python main.py --dataset 'kuzushiji' --model 'cnn' --method 'log_ce' --framework 'one_stage' --adv_epochs 100 --at_lr 0.01 --scheduler 'cosine' --sch_epoch 50 --warmup_epoch 10 --seed 1
```

#### Others

```shell
# Supported Datasets
--dataset - ['mnist', 'kuzushiji', 'fashion', 'cifar10', 'svhn', 'cifar100']
# Complementary Loss Functions
--method - ['free', 'nn', 'ga', 'pc', 'forward', 'scl_exp', 'scl_nl', 'mae', 'mse', 'ce', 'gce', 'phuber_ce', 'log', 'exp', 'l_uw', 'l_w', 'log_ce', 'exp_ce']
# Multiple Complementary Labels (MCLs)
--cl_num - (1-9) the number of complementary labels of each data; (0) MCLs data distribution of ICML2020
```

### Reference

* [NeurIPS 2017] - [Learning from complementary labels](https://arxiv.org/abs/1705.07541)
* [ECCV 2018] - [Learning with biased complementary labels](https://arxiv.org/abs/1711.09535)
* [ICML 2019] - [Complementary-label learning for arbitrary losses and models](https://arxiv.org/abs/1810.04327)
* [ICML 2020] - [Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels](https://arxiv.org/abs/2007.02235)
* [ICML 2020] - [Learning with Multiple Complementary Labels](https://arxiv.org/abs/1912.12927v3)
* [IJCAI 2021] - [Learning from Complementary Labels via Partial-Output Consistency Regularization](https://www.ijcai.org/proceedings/2021/0423.pdf)
* [ICML 2021] - [Discriminative Complementary-Label Learning with Weighted Loss](http://proceedings.mlr.press/v139/gao21d/gao21d.pdf)

### Acknowledgments

Thank the authors of *"Complementary-label learning for arbitrary losses and models"* for the open-source [code](https://github.com/takashiishida/comp) and issue discussion. Other codebases may be found on the corresponding author's homepage.
