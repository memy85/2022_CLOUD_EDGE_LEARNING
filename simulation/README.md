# Create synthetic data-set for a simple simulation

## Edge dataset

2-dimensional dataset with (num_client, num_sample_per_client)

$$
D_{e}=(sin(x), cos(x)) \in \R^{(num_client, num_sample_per_client, 2)}

$$

## Cloud dataset

2-dimensional dataset with (num_client, num_sample_per_client)

$$
D_{c}=(3x+2, 5x^{2}+4x) \in \R^{(num_client, num_sample_per_client, 2)}

$$

## label

$$
y = D_{e} @ D_{c} + e \\

$$

where e is error term ~ N(0, 0.1)

@ is element-wise multiplication

== y 원소 1개에 대해서

(5x+1)(x+1)(sinx+cosx)+e
