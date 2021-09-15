## AT with Complementary Labels

### Run
The following demo will show the results with the MNIST dataset.  After running the code, you should see a text file with the results saved in the same directory.  The results will have three columns: epoch number, training accuracy, and test accuracy.

```bash
python main.py -h
```

#### Methods and models
In `demo.py`, specify the `method` argument to choose one of the 5 methods available:

- `ga`: Gradient ascent version (Algorithm 1) in [1].
- `nn`: Non-negative risk estimator with the max operator in [1].
- `free`: Assumption-free risk estimator based on Theorem 1 in [1].
- `forward`: Forward correction method in [2].
- `pc`: Pairwise comparison with sigmoid loss in [3].

Specify the `model` argument:

- `linear`: Linear model
- `mlp`: Multi-layer perceptron with one hidden layer (500 units)

#### Results on CIFAR-10 (CL) - [Logs](https://drive.google.com/drive/folders/1EhzJDNdAbWm6yGQ8yev128leVsjXji3p?usp=sharing)

> Settings: For CIFAR-10, ResNet-34 was used with weight decay of 5e−4 and initial learning rate of 1e−2. For optimization, SGD was used with the momentum set to 0.9. Learning rate was halved every 30 epochs. We train the model for 300 epochs with batch_size = 256.

|             CL              |  nature_test_acc  |
| :-------------------------: | :---------------: |
| ~~unbiased_risk_estimator~~ |  ~~9.29 / 26.2~~  |
|      free (Ishida 19)       |   11.61 / 29.15   |
|       nn (Ishida 19)        |   23.78 / 35.27   |
|       ga (Ishida 19)        |   31.43 / 31.67   |
|      ~~modified_free~~      | ~~11.68 / 24.00~~ |
|       ~~modified_nn~~       | ~~22.50 / 34.73~~ |
|       ~~modified_ga~~       | ~~26.91 / 27.42~~ |
|         ~~scl_exp~~         | ~~10.00 / 10.00~~ |



|           **AT**            |                   |
| :-------------------------: | :---------------: |
|      adv_min_nn_max_nn      |                   |
|      adv_min_nn_min_ce      |                   |
|    adv_min_free_max_free    |                   |
