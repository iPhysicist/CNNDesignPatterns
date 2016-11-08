# CNNDesignPatterns

This repository hosts the Caffe code and prototxt files for the CNN Design Patterns paper available at https://arxiv.org/abs/1611.00847.  If you use this work in your research, please cite this paper.


In order to use the prototxt files, you should install Caffe along with the fractalnet code located at https://github.com/gustavla/fractalnet/tree/master/caffe.

To use the SBN and TSN prototxt files, you will need to also install the freeze-drop-path code and the instructions are below.

Prototxt files
==============

    solver.prototxt: The solver file.
    fractalnet.prototxt: The original fractalnet architecture
    fractalnet-AvgPool.prototxt: Replaces max pooling with average pooling in the original fractalnet architecture
    FoF.prototxt: The fractal of fractalnet architecture.
    FoF-TSN.prototxt: Taylor series modification where branch 2 is squared and branch 3 cubed prior to the final fractal join.
    SBN-FDP.prototxt: Stagewise Boosting network architecture with freeze-drop-path as the final join.
    TSN-FDP.prototxt: Taylor series modification of the SBN architecture.


Freeze-drop-path
================

Copy the hpp file into your include/caffe/layers folder and copy the cpp and cu files into your src/caffe/layers folder. 

Then, add the following to your ``src/caffe/proto/caffe.proto`` file in ``LayerParameter``:

  optional FreezeDropPathParameter freeze_drop_path_param = 148;

Set ``148`` to whatever you want that is not in conflict with another layer's parameters. Also add the following to the bottom ``caffe.proto``:

    message FreezeDropPathParameter {
      optional uint32 num_iter_per_cycle = 1 [default = 0];
      optional uint32 interval_type = 2 [default = 0];
    }

Re-compile and you should now have access to the ``freezedroppath`` unit.

Usage
-----
Here is an example of how to use freeze-drop-path with two layers, with the stochastic option and intervals that increase with the square of the branch number, respectively::

    layer {
      name: "freeze_drop_path"
      type: "FreezeDropPath"
      bottom: "pool2_27_plus"
      bottom: "extra_mid_join2a"
      top: "extra_mid_join2"
      freeze_drop_path_param {
        num_iter_per_cycle: 0
        interval_type: 0
      }
    }

If you want to use freeze-drop-path deterministically, set num_iter_per_cycle to the number of iterations for a cycle through the all the branches.

