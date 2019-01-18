# Cas2Vec
Implementation of the Cas2Vec algorithm as described in the paper, [Network-Agnostic Cascade Prediction in Online Social Networks](https://ieeexplore.ieee.org/document/8554730). 
In this paper we investigate how one can effectively do virality prediction without requiring knowledge of the underlying nework. 
The complete detail is given in our paper. 
The source code contains minor changes for some terms used in the paper and we have included an inline documentation for such changes. 
Eg. slices are named as bins.
### Requirements!
  - Tensorflow 1.5+
  - Numpy
## Usage
#### Example usage
```sh
$ python cas2vec/main.py --cas-path data/train.txt
```

#### Input format
Each Line is assoiated with a cascade id followed by a sequence of infection events separated by white space. 
An infection event contains a pair (node_id,timestamp), timestamp should be relative and in seconds. 
That is, the timestamp of the first event is 0, and for the rest the time stamps are simply the difference 
(in seconds) between the current event the previous event.
For example:

1 1000,0 47,890 2,1000, 700,3808, 475,20070

2 3290,0 9798,10 90,15, 98,16, 987,20

### Possible Parameters


>`--cas-path:`
A path to cascade file. Default is empty string.

>`--model-dir:`
A path to a directory to save a trained model. Default is ```./models```

>`--obs-time:`
An observation time. Default is 1 hour

>`--prd-time:`
A Prediction time (it should be greater than --obs-time, i.e. --obs-time + delta, delta > 0). 
Default is 16 hours

>`--time-unit:`
The time unit used to specify the previous two parameters (possible values are, h for hour, m for minute, and s for second). Default is 'h'

>`--disc-method:`
 The discretization method to be used possible values are (counter, const). Default is counter.

>`--num-bins:`
The number of bins (slices). Default is 40

>`--seq-len:`
Sequence length, relevant when the `--disc-method` is 'const'. Default is 100

>`--threshold:`
A virality threshold. Default is 1000

>`--dev-ratio:`
The fraction of points to be used as a development set. Default is 0.2. Set it to 0 during test.

>`--kernel-size:`
A list of kerne sizes to be used in the `Convolutional Layer`. Default is empty, but internally it will be set to -- [3, 4]

>`--filters:`
A list of numbers associated with the number of filters for each kernel size in the above argument. Default is empty, but internally it will be set to -- [32, 16].

>`--fcc-layers:`. 
A list containing the configuration for the `Fully Connected Layer`, which comes after the `Convolutional Layer` and before the `Prediction Layer`. Default is empty, but internally it is set to -- [512, 128].

>`--size:`
The embedding size related to the `Cascade Embedding Matrix`. Default is 128.

>`--lr:`
Learning rate. Default is 0.0001

>`--batch-size:`.
Batch size. Default is 32

>`--epochs:`
Number of epochs. Default is 10

>`--sf:`
Sampling factor used to control the number of non-viral cascades.  Default is 1. The number of non-vrial cascades is proportional to the number of viral cascades by a factor `--sf`


Citing
------

If you find Cas2Vec useful in your research, we kindly ask that you cite the following paper:
```
@INPROCEEDINGS{8554730,
author={Zekraias T. Kefato and N. Sheikh and L. Bahri and A. Soliman and A. Montresor and S. Girdzijauskas},
booktitle={2018 Fifth International Conference on Social Networks Analysis, Management and Security (SNAMS)},
title={CAS2VEC: Network-Agnostic Cascade Prediction in Online Social Networks},
year={2018},
publisher = {IEEE},
pages={72-79},
keywords={Cascade Prediction, Convolutional Neural Networks, Deep Learning, Social Networks},
doi={10.1109/SNAMS.2018.8554730},
month={Oct},}
```