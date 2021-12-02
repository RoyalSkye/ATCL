## Learning with Complementary Labels

### Run
```shell
# EXP
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset 'kuzushiji' --cl_num=1 --method 'exp' 2>&1 &
# EXP+AT
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset 'kuzushiji' --cl_num=1 --method 'exp' --at 2>&1 &
```

### Reference

1. Y. T. Chou, G. Niu, H. T. Lin, and M. Sugiyama.<br>**Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels**.<br>In *ICML 2020*. [[paper]](https://arxiv.org/abs/2007.02235)
2. T. Ishida, G. Niu, A. K. Menon, and M. Sugiyama.<br>**Complementary-label learning for arbitrary losses and models**.<br>In *ICML 2019*. [[paper]](https://arxiv.org/abs/1810.04327)
3. Yu, X., Liu, T., Gong, M., and Tao, D.<br>**Learning with biased complementary labels**.<br>In *ECCV 2018*. [[paper]](https://arxiv.org/abs/1711.09535)
4. T. Ishida, G. Niu, W. Hu, and M. Sugiyama.<br>**Learning from complementary labels**.<br>In *NeurIPS 2017*. [[paper]](https://arxiv.org/abs/1705.07541)

