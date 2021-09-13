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

```shell
dYAfs2Dm

# ATCL - unbiased_risk_estimator / ga

# ATCL1 - assump_free_loss / nn

# ATCL2 - min ure_nn max ure_nn
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py 2>&1 &
# ATCL3 - min ure_nn min ce
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --loss "biased" 2>&1 &
```

