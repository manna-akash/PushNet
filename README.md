# Push-Net: Deep Recurrent Neual Network for Planar Pushing Objects of Unknown Physical Properties

## What is Push-Net ?
* a deep recurrent neural network that selects actions to push objects with unknown phyical properties
* unknown physical properties: center of mass, mass distribution, friction coefficients etc.
* for technical details, refer to the [paper](http://motion.comp.nus.edu.sg/wp-content/uploads/2018/06/rss18push.pdf)

## Environment and Dependencies
* Ubuntu 14.04
* Python 2.7.6
* Pytorch 0.3.1
* GPU: GTX 980M
* CUDA version: 8.0.44
* [imutils](https://github.com/jrosebr1/imutils)

## Usage
* Input: an current input image mask of size 128 x 106, and goal specification (see [push_net_main.py](push_net_main.py))
* Output: the best push action on the image plane
* Example:
  
  input image : ```test.jpg```
  
  ```python train.py```
  
  







