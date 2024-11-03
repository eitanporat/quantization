# Benchmarking Quantization Algorithms

In this project, I compared the accuracy of neural networks of different QAT techniques. 
From preliminary analysis the randomized permutation linear layer achieves better performance than naive quantization and hadamard quantization.
I didn't add any subsequent codebook quantization as in QuIP# paper: https://arxiv.org/abs/2402.04396.

Results are detailed [here](https://github.com/eitanporat/quantization/blob/main/results_vit_imagenet.csv)
