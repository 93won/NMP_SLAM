# Efficient and Robust Multi-Modal Belief Propagation SLAM Using Clusrering-Based Re-Parameterization

Implementation of multi-modal belief propagation using clusrering-based re-parameterization for SLAM back-end

Through this SLAM back-end algorithm, robust and efficient optimization results can be obtained for multi-modal factor graphs including false loop closures in environments where structural ambiguities exist.

## Example Codes

### 1-D Example

1-D example implemented in 1d_Simulation.py is an example of constructing and optimizing a multi-modal factor fraph for data association problem as follows:

![1](https://user-images.githubusercontent.com/38591115/109464852-ae2ac680-7aaa-11eb-8fa2-3d2e956fbe4e.PNG)

![2](https://user-images.githubusercontent.com/38591115/109464873-b551d480-7aaa-11eb-8e4f-9ebf462d8304.PNG)

![3](https://user-images.githubusercontent.com/38591115/109464883-baaf1f00-7aaa-11eb-8da4-b07b76d8e22b.PNG)

![4](https://user-images.githubusercontent.com/38591115/109464891-be42a600-7aaa-11eb-9a24-93c4312c5834.PNG)

![5](https://user-images.githubusercontent.com/38591115/109464894-c13d9680-7aaa-11eb-884f-6eaf9e4b6e98.PNG)

This example is deeply inspired by 1-D example of Reference [2]

### 2-D Example

2-D example implemented in 2d_Simulation.py is an example of 2D simulation

![2d_example](https://user-images.githubusercontent.com/38591115/109468148-8d18a480-7aaf-11eb-97f7-0f380d6f1a9a.png)

In simulation often exists Euclidean distance error between the (a)actual position and (b)the estimated position after a long operation of the robot. If the sensor data set which is obtained in the state of (b) is similar with those obtained in the previous positions (c) and (d), and the distance between the two states of (c) and (d) is in the range of accumulated error, it is often a source of errors in previous SLAM. The proposed algorithm can handle data where such error from structural ambiguities by modeling both in a loop closure factor using a weighted Gaussian mixture.


# TO DO

#### (1) C++ implementation for speed up
#### (2) Implementation of fast triangulation algorithm (LEX_M -> ???)

## Reference

[1] Seungwon Choi and Taewan Kim. (In Review) Efficient and Robust Multi-Modal Belief Propagation SLAM Using Clusrering-Based Re-Parameterization

[2] Dehann Fourie, John Leonard and Michael Kaess (2016) A Nonparametric Belief Solution to the Bayes Tree (IROS)
