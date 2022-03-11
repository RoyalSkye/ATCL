## Adversarial Training with Complementary Labels

### Run

```shell
# For MNIST/Fashion/KMNIST
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset 'kuzushiji' --model 'cnn' --method 'exp' --adv_epochs 100 --at_lr 0.01 --scheduler 'cosine' --sch_epoch 50 2>&1 & 
```

#### Reference

1. Gao, Y., & Zhang, M. L.<br>**Discriminative Complementary-Label Learning with Weighted Loss**<br>In *ICML 2021*. [[paper]](http://proceedings.mlr.press/v139/gao21d/gao21d.pdf)
2. Feng, L., Kaneko, T., Han, B., Niu, G., An, B., & Sugiyama, M.<br>**Learning with Multiple Complementary Labels**<br>In *ICML 2020*. [[paper]](https://arxiv.org/abs/1912.12927v3)
1. Chou, Y. T., Niu, G., Lin, H. T., & Sugiyama, M.<br>**Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels**<br>In *ICML 2020*. [[paper]](https://arxiv.org/abs/2007.02235)
2. Ishida, T., Niu, G., Menon, A., & Sugiyama, M.<br>**Complementary-label learning for arbitrary losses and models**<br>In *ICML 2019*. [[paper]](https://arxiv.org/abs/1810.04327)
3. Yu, X., Liu, T., Gong, M., & Tao, D.<br>**Learning with biased complementary labels**<br>In *ECCV 2018*. [[paper]](https://arxiv.org/abs/1711.09535)
4. Ishida, T., Niu, G., Hu, W., & Sugiyama, M.<br>**Learning from complementary labels**<br>In *NeurIPS 2017*. [[paper]](https://arxiv.org/abs/1705.07541)

