# Efficient and Robust Multi-Modal Belief Propagation SLAM Using Clusrering-Based Re-Parameterization

Implementation of multi-modal belief propagation using clusrering-based re-parameterization for SLAM back-end

Through this SLAM back-end algorithm, robust and efficient optimization results can be obtained for multi-modal factor graphs including false loop closures in environments where structural ambiguities exist.

## Example Codes

### 1-D Example

The 1-D example implemented in 1d_Simulation.py is an example of constructing and optimizing a multi-modal factor fraph for data association problem as follows:

![1](https://user-images.githubusercontent.com/38591115/109464852-ae2ac680-7aaa-11eb-8fa2-3d2e956fbe4e.PNG)

![2](https://user-images.githubusercontent.com/38591115/109464873-b551d480-7aaa-11eb-8e4f-9ebf462d8304.PNG)

![3](https://user-images.githubusercontent.com/38591115/109464883-baaf1f00-7aaa-11eb-8da4-b07b76d8e22b.PNG)

![4](https://user-images.githubusercontent.com/38591115/109464891-be42a600-7aaa-11eb-9a24-93c4312c5834.PNG)

![5](https://user-images.githubusercontent.com/38591115/109464894-c13d9680-7aaa-11eb-884f-6eaf9e4b6e98.PNG)

This example is deeply inspired by 1-D example of Reference [2]



## Reference

[1] Seungwon Choi and Taewan Kim. (In Review) Efficient and Robust Multi-Modal Belief Propagation SLAM Using Clusrering-Based Re-Parameterization

[2] Dehann Fourie, John Leonard and Michael Kaess (2016) A Nonparametric Belief Solution to the Bayes Tree (IROS)
