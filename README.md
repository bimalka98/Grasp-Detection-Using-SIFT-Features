# Grasp-Detection 🌱

- Inspired by the paper : [***Grasp Detection under Occlusions Using SIFT Features***](https://www.hindawi.com/journals/complexity/2021/7619794/)
- Tested on:
    * *OpenCV version :  4.5.2*
    * *Numpy version :  1.20.2*

## ⭐ Procedure 
1. Get the 2 grasp points (for a parallel plate gripper) from the user as two mouse clicked points on the template image.
2. Extract all the matching SIFT features from the template and the scene image.
3. Filter out incorrect matches using the method 1 explained in the paper mentioned below (*SIFT Feature Point Matching Based on Improved RANSAC Algorithm*).
4. Calculate the homography transformation matrix using the correctly matched point correspondances.
5. Transform two grasp points on template image on to the scene image.

## ⭐ RANSAC Algorithm performs poorly when there are large number of outliers

```
Homography Matrix :
 [[-1.52476214e+00 -1.04132676e+00  2.52647538e+02]
 [ 1.71652438e-01  1.28717451e-01 -3.49052204e+01]
 [-6.18710693e-03 -4.20617541e-03  1.00000000e+00]]
 
Transformed Grasp Locations :
 [[ 365.94464164 -138.84489806]
 [ 154.21146108   56.54208207]]
```

<img src="figures/RANSAC.png" width="700" />

## ⭐ Improve performance of the RANSAC

Using the method 1 explained in the following paper:

 * ***SIFT Feature Point Matching Based on Improved RANSAC Algorithm*** by Guangjun Shi, Xiangyang Xu, Yaping Dai

Method 2 explained in the paper is not implemented in this code. Because it is not possible to use the method 2 in our scenario where images may be rotated by a large angle. Therefore, cross points are a natural thing in our case.

<img src="figures/iRANSAC.png" width="700" />

**Note**: *However, this improved RANSAC algorithm was also not met the requirements of the project due to its unreliable behavior on the test images. Sometimes it gave accurate results but sometimes it gave inaccurate results. In addition to that, sometimes the matching points given by it was not enough/ not correct  to determine the homography matrix accurately.*
