# WV View Deep Learning Examples

## Overview

The goal of this repository is to provide example workflows for geospatial deep learning. The primary focus is semantic segmentation for pixel-based classification. However, we will also provide examples demonstrating object detection/instance segmentation and scene classification. We plan to continue to add examples and make updates. This material is associated with the WV View [Geospatial Deep Learning](http://www.wvview.org/course_directory.html) seminar.

WV View is supported by AmericaView and the U.S. Geological Survey under Grant/Cooperative Agreement No. G18AP00077.

The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the opinions or policies of the U.S. Geological Survey. Mention of trade names or commercial products does not constitute their endorsement by the U.S. Geological Survey.

## Folders

1. **PyTorch_SAT6:** Use PyTorch/Python and R to classify aerial image scenes using CNNs. Example data are DeepSat SAT-6 Dataset.
2. **PyTorch_UNet_topoDL:** Use PyTorch/Python and R to extract historic surface mine extents from topographic maps using UNet semantic segmentation. Example dataset is topoDL Dataset. 
3. **PyTorch_DeepLab_Inria:** Use PyTorch/Python and R to extract building footprints using DeepLabv3+ semantic segmentation. Example data are Inria Aerial Image Labeling Dataset.
4. **PyTorch_UNetPP_Landcoverai:** Use PyTorch/Python and R to map general land cover using UNet++ semantic segmentation. Example dataset is LandCover.ai. 
5. **PyTorch_UNetPP_wvlcDL:** Use PyTorch/Python and R to map general land cover with incomplete training data using UNett++ semantic segmentation. Example data are wvlcDL Dataset. 


## Videos

The following videos step through example using ArcGIS Pro without any coding, our PyTorch/Python and R examples, and settting up deep learning environments. 

1. [Set Up ArcGIS Pro for Deep Learning](https://youtu.be/z6PAUzPqSkQ) (Completed)
2. [UNet Semantic Segmentation with ArcGIS Pro (topoDL)](https://youtu.be/4HZ41mFhWws) (Completed)
3. [Object Detect/Instance Segmentation with ArcGIS Pro (vfillDL)](https://youtu.be/b1qddjuhIS0) (Completed)
4. [Set up PyTorch Python environment using Anaconda](https://youtu.be/gEkhiP_oCT4) (Completed)
5. [PyTorch/R CNN Scene Classification (DeepSat SAT-6 Dataset)](https://youtu.be/nmRKUynZnc4) (Completed)
6. [PyTorch/R Semantic Segmentation (topoDL)](https://youtu.be/wtwOSWsZ3xM) (Completed)
7. [Pytorch/R Sematnic Segmentation (Inria)](https://youtu.be/Ac20oEYYdMM) (Completed)
8. [PyTorch/R Semantic Segmentation (LandCover.ai)](https://youtu.be/HxyBvugGqaw) (Completed)
9. [PyTorch/R Semantic Segmentation (wvlcDL)]() (In Progress)

## Data

The datasets used in the examples are linked below. 

1. [topoDL](http://www.wvview.org/research.html)
2. [vfillDL](http://www.wvview.org/research.html)
3. [wvlcDL](http://www.wvview.org/research.html)
4. [SAT-6](https://www.kaggle.com/crawford/deepsat-sat6)
5. [Inria](https://project.inria.fr/aerialimagelabeling/)
6. [LandCover.ai](https://landcover.ai/)

## Papers

The listed papers explain the datasets used in our examples. 

1. Maggiori, E., Tarabalka, Y., Charpiat, G. and Alliez, P., 2017, July. Can semantic labeling methods generalize to any city? the inria aerial image labeling benchmark. In 2017 IEEE International Geoscience and Remote Sensing Symposium (IGARSS) (pp. 3226-3229). IEEE.
2. Boguszewski, A., Batorski, D., Ziemba-Jankowska, N., Zambrzycka, A. and Dziedzic, T., 2020. LandCover. ai: Dataset for Automatic Mapping of Buildings, Woodlands and Water from Aerial Imagery. arXiv preprint arXiv:2005.02264.
3. Basu, S., Ganguly, S., Mukhopadhyay, S., DiBiano, R., Karki, M. and Nemani, R., 2015, November. Deepsat: a learning framework for satellite imagery. In Proceedings of the 23rd SIGSPATIAL international conference on advances in geographic information systems (pp. 1-10).
4. Maxwell, A.E., Bester, M.S., Guillen, L.A., Ramezan, C.A., Carpinello, D.J., Fan, Y., Hartley, F.M., Maynard, S.M. and Pyron, J.L., 2020. Semantic Segmentation Deep Learning for Extracting Surface Mine Extents from Historic Topographic Maps. Remote Sensing, 12(24), p.4145.
5. Maxwell, A.E., Pourmohammadi, P. and Poyner, J.D., 2020. Mapping the topographic features of mining-related valley fills using mask R-CNN deep learning and digital elevation data. Remote Sensing, 12(3), p.547.
6. Maxwell, A.E., Strager, M.P., Warner, T.A., Ramezan, C.A., Morgan, A.N. and Pauley, C.E., 2019. Large-Area, High Spatial Resolution Land Cover Mapping Using Random Forests, GEOBIA, and NAIP Orthophotography: Findings and Recommendations. Remote Sensing, 11(12), p.1409.

## Links

Links to software tools and packages. 

### Deep Learning Python

* [PyTorch](https://pytorch.org/)
* [arcgis.learn module](https://developers.arcgis.com/python/api-reference/arcgis.learn.toc.html)
* [Segmentation Models PyTorch](https://developers.arcgis.com/python/api-reference/arcgis.learn.toc.html)
* [Albumentations](https://albumentations.ai/)
* [opencv-python](https://pypi.org/project/opencv-python/)

### Geospatial Data in Python

* [Rasterio](https://rasterio.readthedocs.io/en/latest/)
* [GeoPandas](https://geopandas.org/)

### Python Data Science
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [SciPy](https://www.scipy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

### R General Data Science

* [tidyverse](https://www.tidyverse.org/)
* [dplyr](https://dplyr.tidyverse.org/)
* [ggplot2](https://ggplot2.tidyverse.org/)
* [caret](https://topepo.github.io/caret/)
* [tidymodels](https://www.tidymodels.org/)
* [yardstick](https://cran.r-project.org/web/packages/yardstick/index.html)
* [rfUtilities](https://cran.r-project.org/web/packages/rfUtilities/index.html)
* [diffeR](https://cran.r-project.org/web/packages/diffeR/index.html)
* [pROC](https://cran.r-project.org/web/packages/pROC/index.html)
* [torch](https://cran.r-project.org/web/packages/torch/index.html)
* [imager](https://dahtah.github.io/imager/)
* [randomForest](https://cran.r-project.org/web/packages/randomForest/index.html)
* [e1071](https://cran.r-project.org/web/packages/e1071/index.html)

### R Geospatial Data Science

* [sp](https://cran.r-project.org/web/packages/sp/index.html)
* [rgdal](https://cran.r-project.org/web/packages/rgdal/index.html)
* [sf](https://cran.r-project.org/web/packages/sf/index.html)
* [star](https://cran.r-project.org/web/packages/stars/index.html)
* [raster](https://cran.r-project.org/web/packages/raster/index.html)
* [terra](https://cran.r-project.org/web/packages/terra/index.html)
* [leaflet](https://rstudio.github.io/leaflet/)
* [tmap](https://cran.r-project.org/web/packages/tmap/vignettes/tmap-getstarted.html)
* [tmaptools](https://cran.r-project.org/web/packages/tmaptools/index.html)
* [lidR](https://cran.r-project.org/web/packages/lidR/index.html)

