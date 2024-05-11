# Surface Classification of Urban Cities using IRIS and Sentinel-2 Data

The purpose of this project is to explore the effectiveness of IRIS (Intelligently Reinforced Image Segmentation) for surface classification using an urban setting. It creates semi-supervised classified imagery that aims to differentiate between green spaces and urban walkways/buildings/etc., in the context of the Gardens by the Bay area of the south-eastern country of Singapore, which will be used as training data to critically analyse the performance of IRIS in comparison to supervised/unsupervised ML methods. The project can be used for a wide range of applications requiring more efficient surface classifications. Future applications include the aerial monitoring of urban spaces to understand how they change and wear over time.

Summary of results: semi-supervised learning produced a more accurate [x value] than the supervised/unsupervised ML method of surface classification.

This repository provides access to the relevant code, ipynb files, figures and version histories for the project.

# List of contents
* Background context
* Methodology
* Results and Discussion
* References

# Background context

* Urban greenery, its Importance and Introduction to The Project Location

Urban green spaces comprise pedestrian pathways, vehicle roadways, buildings, and patches of vegetation and water. Urban green spaces can be complex to classify if pedestrian pathways are woven around naturalistic designs often featuring variable topographies and towering skyscrapers. These objects can create shadows or complex visual features that make green spaces and roads difficult to distinguish. The use of machine learning that leverages satellite imagery could help to prevent accidents and could increase the efficiency of using automated vehicles to navigate roadways that rely on automated navigation systems. Driverless vehicle technology is currently facing significant investment in the south-eastern island country of Singapore (Ng and Kim, 2021).

The Gardens by the Bay, located in Marina Bays area of Singapore, is home to conservation projects, tourist destinations, musuems and more. It serves to foster biodiverse cityspaces and to educate people about sustainability, benefitting the physical and mental health of locals and visitors. It is an important economic source that generates revenue from its annual 45 million visitors and tourists (Yale Center for Business and the Environment, 2024). Having been a host to many redevelopment projects over the last decade, it has undergone fast change in its land surface and coastline and it faces challenges such as erosion and land subsidence due to the rising sea level associated with climate change (Bai et al., 2023). Some key features of the area include:
* Trees, grass, bushes, shrubs and reeds: catagorised in this project as 'green spaces'.
* Artificial Lakes: hosting a range of insects and birds such as dragonflies and hummingbirds. Lakes are often inaccessible by vehicles and people, so from now on they will be catagorised within 'green spaces'.
* Buildings: comprising residential areas, offices, shopping malls, hotels and various tourist attractions such as museums, glasshouses and the famous artificial 'Supertrees'.
* Roads: for the daily movement of traffic or for industrial purposes such as construction.

<img width="952" alt="Screenshot 2024-05-11 at 15 29 36" src="https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/87e0e140-3a42-4fdb-911f-37b6797a977d">
<img width="1053" alt="Screenshot 2024-05-11 at 15 31 44" src="https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/9a4529ba-b644-4488-b219-644f0f5103b8">
![image](https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/2320e58a-1c44-474b-a1b6-d9917c47d4af)

Evaluate increased foliage during summer that may obscure roads and create more shadows compared to winter satellite images.

* Copernicus, Sentinel-2 and IRIS
  
Copernicus is a database platform that provides free access to satellite data and imagery that is close to real time. It was developed by the ESA-Phil lab, is managed by the European Commision and is contributed to by the Euporean Space Agency (ESA) and the European Environmental Agency (EEA). Currently it retrieves data from a satellite constellation series mostly comprised of the Sentinel satellites to monitor air and water quality and to observe land use change, climate change and nataural disasters.
Sentinel-2 is
IRIS is a recent annotation tool developed over the past 4 years and aimed to make surface classification of multispectral and multimodel imagery a quicker and easier process for users. It leverages the iterative and sequential machine learning technique of artificial intelligence known as 'gradient boosted decision trees'. Each decision tree identifies errors from the previous tree to make corrections via a sequential manner. It can run on Linux, Windows and Mac OS (Wheeler, 2024).

* AI Background
Introduce unsupervised, K-Means clustering

# Methodology:

Step 1: Data Collection

The image data of Sentinel-2 can be retrieved using Copernicus v1.10.1.

Using Copernicus, draw a polygon using the ? function to select the area of interest. For the coorodinates [name them], 4 co-located images from the satellite were found.

<img width="1036" alt="Screenshot 2024-05-10 at 20 04 43" src="https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/15694c72-e2c2-45a3-9f91-bc677c89fcd5">

Step 2: Docker was used to open the IRIS framework

Installing Iris directly from the GitHub repository can be time-consuming and complicated, because of the manual installation of dependencies and configuration steps. Users of the 'Docker' software can quickly bring IRIS into a centralised environment by streamlining the deployment and running process, fostering collaboration across multiple devices.

Download Docker: https://docs.docker.com/get-docker/

Create a Docker Account (Optional): https://hub.docker.com/ (this step for those who may want to manage large sets of images and repositories via Docker Hub)

To create ingest satellite imagery within the IRIS framework, pull the image via the Mac OS's Terminal application. 

Step 3: Training using IRIS semi-supervised classification

Using IRIS, green spaces were partially optically catagorised (~2% of the whole image was supervised training data, whereas the rest of the image was identified using IRIS' artificial intelligence system). This resulted in a semi-supervised classification. 

Step 4: Model comparison with ML techniques

Unsupervised K-means clustering 

Step 6: Model evaluation

The models will be assessed for overfitting and underfitting 

The metric to assess this will be a loss function, also known as a cost function, which quantifies how distant a model’s predictions are from the actual values. Minimising this value of the deviation from the actual results is preferred. Common loss functions include Mean Squared Error for regression tasks and Cross-Entropy for classification tasks.

# Results and Discussion

After completing Steps 1-5, the resulting masks can be found below
[screenshot]

Interpretation: In addition to using IRIS the student would need to develop a notebook describing their masked images in the context of the input features and compare with other unsupervised or supervised ML techniques.

After completing Steps 6- ,

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

Bai, Z., Wang, Y., Li, M., Sun, Y., Zhang, X., Wu, Y., Li, Y. and Li, D., 2023. Land Subsidence in the Singapore Coastal Area with Long Time Series of TerraSAR-X SAR Data. Remote Sensing, 15(9), pp.2415.

Ng, V. and Kim, H.M., 2021. Autonomous vehicles and smart cities: A case study of Singapore. In Smart cities for technological and social innovation. Academic Press, pp. 265-287.

Wheeler, J. (2024). ESA-PhiLab/iris. [online] Available at: https://github.com/ESA-PhiLab/iris [Accessed 11 May 2024].

‌Yale Center for Business and the Environment. (2024). Marina Bay Sands. [online] Available at: https://cbey.yale.edu/research/marina-bay-sands#:~:text=Standing%20at%20water [Accessed May 2024].

‌

  
3. Seeger, M. (2004). 'Gaussian processes for machine learning'. International journal of neural systems, 14(02), pp.69-106.
4. Oyekola, M.A. and Adewuyi, G.K. (2018). 'Unsupervised classification in land cover types using remote sensing and GIS techniques'. International Journal of Science and Engineering Investigations, 7(72), pp.11-18.
5. Patro, R.N., Subudhi, S., Biswal, P.K. and Dell’acqua, F. (2021). 'A review of unsupervised band selection techniques: Land cover classification for hyperspectral earth observation data'. IEEE Geoscience and Remote Sensing Magazine, 9(3), pp.72-111.
6. Camps-Valls, G., Martino, L., Svendsen, D.H., Campos-Taberner, M., Muñoz-Marí, J., Laparra, V., Luengo, D. and García-Haro, F.J. (2018). 'Physics-aware Gaussian processes in remote sensing'. Applied Soft Computing, 68, pp.69-82.
7. Chen, W., Tsamados, M., Willatt, R., Brockley, D., Deisenroth, M., De Rijke-Thomas, C., Francis, A., Hirata, L., Johnson, T., Lawrence, I. and Landy, J. (2024). 'Co-located OLCI optical imagery and SAR altimetry from Sentinel-3 for enhanced surface classification in sea ice (No. EGU24-9175)'. Copernicus Meetings.
8. Wang, Q., Shi, W., Li, Z. and Atkinson, P.M. (2016). 'Fusion of Sentinel-2 images'. Remote sensing of environment, 187, pp.241-252.
