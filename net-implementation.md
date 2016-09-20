# Implementing the neural net

The network is fully connected, with binarized weights and neuron outputs (terminology?),
ie. they are either `1` or `-1`. 
Calculating output for a single neuron is vector inner product. where the vectors are
the outputs of the neurons in the previous layer, and the weights of the edges into the neuron.

As the weights are either `1` or `-1`, a multiplication becomes _equivalence_ (or negated XOR) when we encode `1` as `1`, and `-1` as `0`: 

| x | y | x = y |
| --- | --- | ------ |
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

I am not sure what can be done about the summing, as we also need to offset the result with some constant, as Ulrich describes 
[here](https://github.com/Ytelse/Design/blob/master/network_simplification/network_simplification.pdf).
We can floor the `b/a` constant, so we only work with integers, but since we don't know the sign of the inner product,
further shortcuts sounds hard.

This is a suggestion for how to implement the net.
Let `w_i` denote the weight from neuron `i` in the previous layer, and `x_i` denote the output of the same neuron.

```
w_1  x_1       w_2  x_2       w_3  x_3       w_4  x_4
 |    |         |    |         |    |         |    |
   eqv            eqv            eqv            eqv
    |              |              |              |
    |              |   __________/               |
    \____________  |  / _________________________/
                 | | | |
                  Sum
                   |
                   |  ____  b/a
                   | | 
                   Sub
                   |
                   |
                  msb
                   |
                  not
                   |
```

By selecting the `MSB` and `NOT`ing it, we get `0` on negative numbers, and `1` on positive numbers.

Somehow, the `Sum` block must handle the fact that `0` is actually `-1`. We _could_ add as usual, multiply by two, and subtract width,
(1024 for the large net), but this may be too much logic, if we want to minimize the size of the net (which we do?)

## About the constant

By checking the 60000 `b/a` constants of the large neural net, we can see that they range from `-253` to `486`, so we would need about `10` bytes to represent every number. However, [this](https://github.com/Ytelse/NeuralNet/commit/911dacbe9f636a7b6a82c62bab016ef901f51f35) commit shows in the Rust implementation, clamping the constants to 8-bit signed numbers only decreases the hit rate of the net by 0.05% (from 97.33% to 97.28%). Therefore, I think we can assume that each parameter can safely be `ceil`ed to an int, and clamped to signed 8bit.
