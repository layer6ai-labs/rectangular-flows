# It is not recommended to actually run this full script since it will take a long time.
# We just demonstrate below the commands to modify hyperparameters and obtain the image runs.

# MNIST, RNFs-ML (exact)
(
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --model non-square --dataset mnist --config log_jacobian_method=cholesky
    --config regularization_param=5 --config prior_num_density_layers=10
)

# MNIST, RNFs-ML (K=1)
(
    CUDA_VISIBLE_DEVICES=0,1 ./main.py --model non-square --dataset mnist --config log_jacobian_method=hutch_with_cg
    --config regularization_param=5 --config cg_tolerance=0.001 --config prior_num_density_layers=10
)

# MNIST, RNFs-ML (K=4)
(
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --model non-square --dataset mnist --config log_jacobian_method=hutch_with_cg
    --config regularization_param=50 --config prior_num_density_layers=5 --config hutchinson_samples=4
)

# MNIST, RNFs-TS
CUDA_VISIBLE_DEVICES=0,1 ./main.py --model non-square --baseline --dataset mnist --config regularization_param=50 --config prior_num_density_layers=5

##############################################

# Fashion-MNIST, RNFs-ML (exact)
(
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --model non-square --dataset fashion-mnist --config log_jacobian_method=cholesky
    --config regularization_param=50 --config prior_num_density_layers=10
)

# Fashion-MNIST, RNFs-ML (K=1)
(
    CUDA_VISIBLE_DEVICES=0,1 ./main.py --model non-square --dataset fashion-mnist --config log_jacobian_method=hutch_with_cg
    --config regularization_param=50 --config hutchinson_distribution=rademacher --config prior_num_density_layers=5
)

# Fashion-MNIST, RNFs-ML (K=4)
(
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --model non-square --dataset fashion-mnist --config log_jacobian_method=hutch_with_cg
    --config regularization_param=50 --config hutchinson_distribution=rademacher --config prior_num_density_layers=10
    --config hutchinson_samples=4
)

# Fashion-MNIST, RNFs-TS
(
    CUDA_VISIBLE_DEVICES=0,1 ./main.py --model non-square --baseline --dataset fashion-mnist --config regularization_param=5
    --config prior_num_density_layers=10 --config likelihood_warmup=False --config g_hidden_channels=[64,64,64,64]
)

##############################################

# CIFAR10, RNFs-ML (exact)
(
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./main.py --model non-square --dataset cifar10 --config log_jacobian_method=cholesky
    --config regularization_param=5 --config prior_num_density_layers=10
)

# CIFAR10, RNFs-TS
(
    CUDA_VISIBLE_DEVICES=0,1 ./main.py --model non-square --dataset cifar10 --baseline
    --config regularization_param=5 --config prior_num_density_layers=10
)
