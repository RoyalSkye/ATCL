## Adversarial Training with Complementary Labels

### Setup

* Python 3.8
* [PyTorch 1.11.0](https://pytorch.org)
* [Autoattack](https://github.com/fra31/auto-attack)

### Run

```shell
# For MNIST/Fashion/KMNIST
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset 'mnist' --model 'cnn' --method 'log_ce' --framework 'one_stage' --adv_epochs 100 --at_lr 0.01 --scheduler 'cosine' --sch_epoch 50 --warmup_epoch 10 --out_dir ./mnist_cnn_log_ce_10_cos_50_0.01 2>&1 &
# For CIFAR10
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset 'cifar10' --model 'resnet18' --method 'log_ce' --framework 'one_stage' --adv_epochs 120 --at_lr 0.01 --scheduler 'cosine' --sch_epoch 30 --warmup_epoch 50 --out_dir ./cifar10_resnet18_log_ce_50_cos_30_0.01 2>&1 &
# Two_stage
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset 'mnist' --model 'cnn' --method 'log' --framework 'two_stage' --cl_epochs 50 --adv_epochs 50 --cl_lr 0.01 --at_lr 0.01 --out_dir ./mnist_cnn_two_stage_0.01_0.01 2>&1 & 
```

#### Reference

1. Ishida, T., Niu, G., Hu, W., & Sugiyama, M.<br>**Learning from complementary labels**<br>In *NeurIPS 2017*. [[paper]](https://arxiv.org/abs/1705.07541)
1. Yu, X., Liu, T., Gong, M., & Tao, D.<br>**Learning with biased complementary labels**<br>In *ECCV 2018*. [[paper]](https://arxiv.org/abs/1711.09535)
1. Ishida, T., Niu, G., Menon, A., & Sugiyama, M.<br>**Complementary-label learning for arbitrary losses and models**<br>In *ICML 2019*. [[paper]](https://arxiv.org/abs/1810.04327)
1. Chou, Y. T., Niu, G., Lin, H. T., & Sugiyama, M.<br>**Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels**<br>In *ICML 2020*. [[paper]](https://arxiv.org/abs/2007.02235)
1. Feng, L., Kaneko, T., Han, B., Niu, G., An, B., & Sugiyama, M.<br>**Learning with Multiple Complementary Labels**<br>In *ICML 2020*. [[paper]](https://arxiv.org/abs/1912.12927v3)
1. Wang, D. B., Feng, L., & Zhang, M. L.<br>**Learning from Complementary Labels via Partial-Output Consistency Regularization**<br>In *IJCAI 2021*. [[paper]](https://www.ijcai.org/proceedings/2021/0423.pdf)
1. Gao, Y., & Zhang, M. L.<br>**Discriminative Complementary-Label Learning with Weighted Loss**<br>In *ICML 2021*. [[paper]](http://proceedings.mlr.press/v139/gao21d/gao21d.pdf)

