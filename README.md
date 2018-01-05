# ExtendedTinyFaces
Analysis and application of the Finding Tiny Faces (P. Hu) [paper](https://arxiv.org/abs/1612.04402).  

### Introduction
The paper - released at CVPR 2017 - deals with finding small objects (particularly faces in our case) in an image, 
based on scale-specific detectors by using features defined over single (deep) feature hierarchy : 
Scale Invariance, Image resolution, Contextual reasoning. The algorithm is based on foveal descriptors, i.e blurring the peripheral image to encode and give just enough 
information about the context, mimicking the human vision.   
The subject is still an open challenge and we would like to enlarge it to new horizons and 
experiment this ap- proach to different applications. The goal would be to deeply understand 
the choices of the paper, together with their applications on subjects related to security and identification.

### Face Recognition
Other applications of the paper would be for face recognition. 
Indeed, we aim at building a Python pipeline for face recognition. 
We would like to use [face alignment](https://pdfs.semanticscholar.org/d78b/6a5b0dcaa81b1faea5fb0000045a62513567.pdf) and 
[face embedding](https://arxiv.org/abs/1503.03832) to achieve face classification.   
The first application, we would like to explore include : counting the many different faces 
(numerous people displayed with different size in the picture) in a video of a crowded public demonstration.  
This application could be fine in this 
[notebook](https://github.com/alexattia/ExtendedTinyFaces/blob/master/Video%20detection.ipynb).
