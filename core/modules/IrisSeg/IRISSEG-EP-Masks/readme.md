Irisi Segmentation Groundtruth Elliptical/Polynomial Dataset (IRISEG-EP)
========================================================================

The content of the zip file should contain the following data:

 * The pregenerated masks for the whole dataset in a _mask_ subdirectory.
 * The software to generate masks and calculate recall, precision and the F-measure contained in a _software_ subdirecty.

If data or software from this package is used please cite the following paper:

__TODO: insert self reference for the paper__

Also see the paper for more information about the iris image databases which were used to generate the ground truth database.

Masks
-----

The masks subdirectory contains the pregenerated masks (using the dataset and manuseg program). The masks are prefixed with the operator name and group by source database.


Software
--------

### Dependencies

 * [OpenCV](http://opencv.org/) version 2.44
 * [Boost](http://www.boost.org/) version 1.53

### Description

This software package contains two programs:

#### manuseg

This is the segmentation program which uses the elliptical and polynomial data and an iris image to extract a mask.

#### maskcmpprf

This is the comparison program for masks, it takes to masks and outputs the precision, recall and F1-measure of the masks.

### Note

The package contains a makefiles for linux and windows. The windows make files uses the [MinGW](http://mingw.org/) environment which is a prerequisite for building the software on windows with the contained makefile. The package also contains pre-compiled windows binaries (32-bit) with required dlls and should be usable in windows without any further dependencies.
