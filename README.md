# Title question

We will use IRIS software for surface classification of SEA ICE/LAND, aiming to create ingest satellite imagery within the IRIS framework to help them create semi-supervised classified imagery. [input the goal sentence]. The purpose of this repository is to provide access to the relevant code, ipynb files, figures and versions for a project as part of an assignment for the module GEOL0069. 

Methodology:

For a supervised approach, students are encouraged to label their own dataset using tools like IRIS.

The goal is to demonstrate the usefulness of IRIS for surface classification on a new type of application and/or satellite imagery (urban setting with Sentinel1 or Sentinel2, cloud detection using Sentinel3, lower latitude examples…).

In addition to using IRIS the student would need to develop a notebook describing their masked images in the context of the input features and compare with other unsupervised or supervised ML techniques.

Data access: See week 3 for fetching satellite imagery from Google EE or Copernicus Dataspace

feedback: Could improve a little on the analysis (alignment) and provide a little more info on the general context (leads and ice) in the README

## Edit from here. Description

You will learn how to find lists of colocated images from pairs of collocated satellites: Sentinel-3 (300m resolution) and Sentinel-2 imagery (10m resolution) as well as collocated altimetry data from Sentinel-3. Unsupervised classification will be performed.

Therefore, this project is divided into two sections:
1) Colocating Sentinel-3 OLCI and Sentinal-2 Optical Data
2) Unsupervised Learning

To successfully classify the echoes in leads and sea ice in this project, you will produce:
1) An average echo shape
2) A standard deviation for these two classes
3) A confusion matrix (to quantify echo classification against the ESA official classification)

![image](https://github.com/MaggieJian/Week4/assets/160494175/3adc0b36-a221-4626-abfb-47120b3ff2f4)
![image](https://github.com/MaggieJian/Week4/assets/160494175/d63adc99-69b7-45ea-9995-e3bd35f0ab4f)

## Getting Started

### Dependencies

1) A Google account
2) A good internet connection for using Google Colab
3) SENTINEL and OLCI files (downloaded and unzipped)
4) Libraries: numpy (as np), ee, os, datetime (timedelta and date), sklearn.cluster (KMeans and DBSCAN), matplotlib.pyplot (as plt), pyproj, shapely.geometry, subprocess, requests, pandas (as pd), rasterio, requests, time, cartopy.crs (as ccrs), sklearn.preprocessing (StandardScaler, MinMaxScaler), sklearn.mixture (GaussianMixture), scipy.cluster.hierarchy (linkage, fcluster), shutil, json, joblib (Parallel), zipfile, sys, glob, netCDF4 (Dataset), scipy.interpolate (griddata), numpy.ma (as ma), glob, matplotlib.patches (Polygon), and scipy.spatial (as spatial and as KDTree).

### Installions

Below are the packages needed for this project

!pip install rasterio
!pip install netCDF4
!pip install basemap
!pip install cartopy

To access the Google Colab file, click https://github.com/MaggieJian/Week4 or go to the GitHub repository (MaggieJian/Week4).

### How to run the program

* Step-by-step bullets
```
code blocks for commands
```

## Troubleshooting

Check: The paths should lead to the corrrect satellite images in your Google Drive. If not, re-upload your unzipped satellite images to your Google Drive or change the path.
```
command to run if program contains helper info
```

## Author contact and help request

Maggie Jian - maggie.jian.21@ucl.ac.uk

Project link - https://github.com/MaggieJian/Week4

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is not licensed

## How to Cite

If you use this code or data in your work, please cite it as:
Jian, M. (2024). Week4. GitHub Repository. https://github.com/MaggieJian/Week4

## Acknowledgments

This project is part of an assignment for the module GEOL0069 (2023/24) taught in UCL Earth Sciences Department

Inspiration, code snippets, etc.
[GEOL0069 Jupyter Notebook] (https://cpomucl.github.io/GEOL0069-AI4EO/intro.html)
[awesome-readme](https://github.com/matiassingers/awesome-readme)
[PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
[dbader](https://github.com/dbader/readme-template)

## References

1. Seeger, M. (2004). 'Gaussian processes for machine learning'. International journal of neural systems, 14(02), pp.69-106.
2. Oyekola, M.A. and Adewuyi, G.K. (2018). 'Unsupervised classification in land cover types using remote sensing and GIS techniques'. International Journal of Science and Engineering Investigations, 7(72), pp.11-18.
3. Patro, R.N., Subudhi, S., Biswal, P.K. and Dell’acqua, F. (2021). 'A review of unsupervised band selection techniques: Land cover classification for hyperspectral earth observation data'. IEEE Geoscience and Remote Sensing Magazine, 9(3), pp.72-111.
4. Camps-Valls, G., Martino, L., Svendsen, D.H., Campos-Taberner, M., Muñoz-Marí, J., Laparra, V., Luengo, D. and García-Haro, F.J. (2018). 'Physics-aware Gaussian processes in remote sensing'. Applied Soft Computing, 68, pp.69-82.
5. Chen, W., Tsamados, M., Willatt, R., Brockley, D., Deisenroth, M., De Rijke-Thomas, C., Francis, A., Hirata, L., Johnson, T., Lawrence, I. and Landy, J. (2024). 'Co-located OLCI optical imagery and SAR altimetry from Sentinel-3 for enhanced surface classification in sea ice (No. EGU24-9175)'. Copernicus Meetings.
6. Wang, Q., Shi, W., Li, Z. and Atkinson, P.M. (2016). 'Fusion of Sentinel-2 images'. Remote sensing of environment, 187, pp.241-252.
