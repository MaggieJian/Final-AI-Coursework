# Surface Classification of Urban Cities using IRIS and Sentinel-2 Data

The purpose of this project is to explore the effectiveness of IRIS (Intelligently Reinforced Image Segmentation) for surface classification using an urban setting. It creates semi-supervised classified imagery that aims to differentiate between green spaces and urban walkways/buildings/etc., in the context of the Gardens by the Bay area of the south-eastern country of Singapore, which will be used as training data to critically analyse the performance of IRIS in comparison to supervised/unsupervised ML methods. The project can be used for a wide range of applications that require more efficient surface classifications. Future applications include the aerial monitoring of urban spaces to understand how they change and wear over time.

Summary of the results: semi-supervised learning produced a more accurate [x value] than the supervised/unsupervised ML method of surface classification, based on a confusion matrix 

This repository provides access to the relevant code, ipynb files, figures and version histories for the project.

# List of contents
* Background context
* Methodology
* Checklist for getting Started
* Results and Discussion
* References

# Background context

* Urban greenery and its importance

Urban green spaces comprise pedestrian pathways, vehicle roadways, buildings, and patches of vegetation and water. Urban green spaces can be complex to classify if pedestrian pathways are woven around naturalistic designs often featuring variable topographies and towering skyscrapers. These objects can create shadows or complex visual features that make green spaces and roads difficult to distinguish. The use of machine learning in satellite imagery analysis could help to prevent accidents and could increase the efficiency of using automated vehicles to navigate roadways that rely on automated navigation systems. Driverless vehicle technology is currently facing significant investment in the south-eastern island country of Singapore (Ng and Kim, 2021).

* Introduction to The Project Location
  
The Gardens by the Bay, located in Marina Bays area of Singapore, is home to conservation projects, workspaces, tourism, museums and more. It is a diverse piece of land that benefits the physical and mental health of locals and visitors because it aims to foster biodiverse cityspaces that educate people about sustainability. The area is also an important economic source that generates significant revenue from its annual 45 million visitors (Yale Center for Business and the Environment, 2024). Having hosted many redevelopment projects over the last decade, it has undergone fast changes to its land surface and its coastline, alonside facing climate challenges such as erosion and land subsidence due to the rising global sea level or the loss of vegetation due to fluctuating rainfall intensity (Bai et al., 2023). Changes to its land features can threaten local species populations, decrease biodiversity and lower the ability of plants to absorb CO2 from the densely populated city.

Key land features that can be visible from satellites include:
* Trees, grass, bushes, shrubs and reeds: catagorised in this project as 'green spaces'.
* Artificial Lakes: also catagorised as 'green spaces', since they host a range of insects, reptiles, mammals and birds (such as dragonflies and hummingbirds) and  lakes are often inaccessible by vehicles and people.
* Buildings: comprising residential areas, offices, shopping malls, hotels and various tourist attractions such as museums, glasshouses and the famous artificial 'Supertrees'. Classified as 'urban'.
* Roads: for the daily movement of traffic or for industrial purposes such as construction. Classified as 'urban'.

<img width="952" alt="Screenshot 2024-05-11 at 15 29 36" src="https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/87e0e140-3a42-4fdb-911f-37b6797a977d">
<img width="1053" alt="Screenshot 2024-05-11 at 15 31 44" src="https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/9a4529ba-b644-4488-b219-644f0f5103b8">

* Copernicus, Sentinel-2 and IRIS
  
Copernicus is a database platform that provides free access to satellite data and imagery that is close to real time. It was developed by the ESA-Phil lab, is managed by the European Commision and is contributed to by the Euporean Space Agency (ESA) and the European Environmental Agency (EEA). Currently it retrieves data from a satellite constellation series mostly comprised of the Sentinel satellites to monitor air and water quality and to observe land use change, climate change and nataural disasters.
Sentinel-2 is
IRIS is a recent annotation tool developed over the past 4 years and aimed to make surface classification of multispectral and multimodel imagery a quicker and easier process for users. It leverages the iterative and sequential machine learning technique of artificial intelligence known as 'gradient boosted decision trees'. Each decision tree identifies errors from the previous tree to make corrections via a sequential manner. It can run on Linux, Windows and Mac OS (Wheeler, 2024).

* Unsupervised learning: K-Means clustering

Introduce unsupervised, K-Means clustering

* Method of evaluation

To compare the performance of IRIS with K, the comparison metrics will be typical of classification models. This includes accuracy, precision, recall, F1 score, support, macro average and weighted average:

![arhg](https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/077f0f41-b650-4d6d-b58a-886fdf12269d)
Image credit: https://en.wikipedia.org/wiki/Precision_and_recall

A confusion matrix will also help to evaluate the performance of IRIS.

# Methodology:

Step 1: Data Collection

The raw image data of Sentinel-2 can be retrieved using Copernicus v1.10.1.

* Using Copernicus, search for the location and filter for Sentinel-2 LC1 by ticking on the boxes in the left.
* Set the dates of interest.
* Select the pencil icon in the top right to draw a polygon around the area of interest.

<img width="1168" alt="Screenshot 2024-05-12 at 13 07 34" src="https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/76c7ef37-ad08-42f6-b668-6b0840328486">

Here is an example of a polygon with 4 collocated satellite images
<img width="1036" alt="Screenshot 2024-05-10 at 20 04 43" src="https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/15694c72-e2c2-45a3-9f91-bc677c89fcd5">

The images retrieved from Copernicus may in .zip format, requiring unzipping via your Mac's Finder application or Windows explorer.

The Copernicus folder that was downloaded for this project is attached in the repository. The JP2 files can also be accessed via this drive: [https://drive.google.com/drive/folders/11u9kGHtYlGD9zqmOu3e1ao4lNWym2Rni?usp=drive_link](URL)

Step 2: Convert Copernicus' .JP2 files to .NPY files

Open the Google Colab file that contains the code to convert the files from .JP2 to .NPY for the next step of this project: [https://drive.google.com/drive/folders/11u9kGHtYlGD9zqmOu3e1ao4lNWym2Rni?usp=drive_link](URL)

Google Colaboratory (Google Colab in short) is

Step 3: Use Docker to open IRIS

Installing Iris directly from the GitHub repository can be time-consuming and complicated, because of the manual installation of dependencies and configuration steps. Instead, users of Docker can quickly bring IRIS into a centralised environment by streamlining the deployment and running processes and fostering collaboration across multiple devices.

Download Docker: https://docs.docker.com/get-docker/

An optional step is to create a Docker Account (this step for those who may want to manage large sets of images and repositories via Docker Hub): https://hub.docker.com/

To pull the IRIS Docker image from Docker Hub, input the following code into your computer terminal or command prompt. 

```
docker pull totony4real/iris:1.0
```

Upon a successful download, the terminal should return a statement of progress
<img width="723" alt="Screenshot 2024-05-12 at 21 39 52" src="https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/49db5b3a-d631-405f-9576-a5a706b6db08">

After completing this step, you can run the Iris Docker container and access the Iris web interface.

Step 4: Training using IRIS semi-supervised classification

Using IRIS, access the downloaded satellite images by running IRIS via Docker.

The .npy files have been stored in the Google Drive in Step 1 ([https://drive.google.com/drive/folders/14fcnGgxK6N6TL7fJLSruuFoRQWwY0y5V?usp=drive_link](URL). Download this folder and upload onto your own Google drive.

Replace `path_to_data` with the new path to your data.

```
docker run -p 80:5000 -v/path_to_data:/dataset/ --rm -it totony4real/iris:1.0
label /dataset/MYCONFIG.json
```
Example: docker run -p 80:5000 -v/Users/maggiejian/Downloads/IRIS_upload_files:/dataset/ --rm -it totony4real/iris:1.0 label /dataset/MYCONFIG.json

The terminal will return a request for you to set an admin password.
![image](https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/e5f3fd31-dd51-4c6d-9e63-17b543a3263c)
After setting an admin password, the following message will be returned on the terminal.
![image](https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/8d755bb6-9c79-4d82-9305-ce629bc51872)
To open IRIS on the web browser, use the following link: http://localhost:80. For example, if the terminal directs you to use http://127.0.0.1:5000, follow http://127.0.0.1:80 instead.
![image](https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/4e84bde0-2b7b-4810-b62a-8f0ba04967e5)
New users will need to register for an IRIS account instead of logging in.
![image](https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/0d0f5946-ec55-4f92-a2a6-386679a5e56d)
Complete the process by choosing and filling in your details. Click Register to be taken to a interface page that is ready with your image to begin your session.

[screenshot]

Tips for using IRIS (Tsamados and Chen, 2022):

**Use the Pencil Tool**
You can use the pencil tool to paint pixels anywhere within the rectangular region marked by red dotted lines. You can zoom in or out of the image by using the mouse scroll wheel or by using two fingers to swipe up or down on the touchpad. This allows you to adjust the image size according to your preference.

**Change the Cursor Size**
To change the cursor size, you can hold the Shift key while using the mouse scroll wheel or swiping up or down with two fingers on the touchpad. This action will increase or decrease the size of the cursor, allowing you to adjust it to your desired size.

**Perform Classification**
Once you have finished painting all the classes needed, IRIS is ready to perform classification and generate output masks.

**Show Drawn Pixels**
You can use the “Show Drawn Pixels” button to display only the pixels that you have drawn. This allows you to see which parts have been classified and processed by IRIS.

**Save the Output Masks**
To save the output masks generated by IRIS, you can click on the ‘Save’ button located at the top of the page. The output masks will be saved in the same folder that contains your data and config file. You can navigate to that folder on your device to locate the saved masks. You are also able to find the corresponding numpy arrays of the saved mask, which may be useful for further analysis.

Mask  creation and download: the coordinates to access the location in this project are provided in the config.json file. EDIT THIS

Using IRIS, green spaces were partially optically catagorised (~2% of the whole image was supervised training data, whereas the rest of the image was identified using IRIS' artificial intelligence system). This resulted in a semi-supervised classification. 

Step 5: Model comparison with ML techniques

Unsupervised K-means clustering 

Step 6: Model evaluation

The models will be assessed for overfitting and underfitting 

The metric to assess this will be a loss function, also known as a cost function, which quantifies how distant a model’s predictions are from the actual values. Minimising this value of the deviation from the actual results is preferred. Common loss functions include Mean Squared Error for regression tasks and Cross-Entropy for classification tasks.


# Checklist for Getting Started

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

Evaluate increased foliage during summer that may obscure roads and create more shadows compared to winter satellite images.

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

# Acknowledgments

This project is part of an assignment for the module GEOL0069 (2023/24) taught in UCL Earth Sciences Department

Inspiration, code snippets, etc.
[GEOL0069 Jupyter Notebook] (https://cpomucl.github.io/GEOL0069-AI4EO/intro.html)
[awesome-readme](https://github.com/matiassingers/awesome-readme)
[PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
[dbader](https://github.com/dbader/readme-template)

# References

Bai, Z., Wang, Y., Li, M., Sun, Y., Zhang, X., Wu, Y., Li, Y. and Li, D., 2023. Land Subsidence in the Singapore Coastal Area with Long Time Series of TerraSAR-X SAR Data. Remote Sensing, 15(9), pp.2415.

Ng, V. and Kim, H.M., 2021. Autonomous vehicles and smart cities: A case study of Singapore. In Smart cities for technological and social innovation. Academic Press, pp. 265-287.

Tsamados, M. and Chen, W. (2022). Introduction to Intelligently Reinforced Image Segmentation (IRIS) — GEOL0069 Guide Book. [online] Available at: https://cpomucl.github.io/GEOL0069-AI4EO/Chapter%201%3AIRIS.html [Accessed 12 May 2024].

Wheeler, J. (2024). ESA-PhiLab/iris. [online] Available at: https://github.com/ESA-PhiLab/iris [Accessed 11 May 2024].

‌Yale Center for Business and the Environment. (2024). Marina Bay Sands. [online] Available at: https://cbey.yale.edu/research/marina-bay-sands#:~:text=Standing%20at%20water [Accessed May 2024].

‌

  
3. Seeger, M. (2004). 'Gaussian processes for machine learning'. International journal of neural systems, 14(02), pp.69-106.
4. Oyekola, M.A. and Adewuyi, G.K. (2018). 'Unsupervised classification in land cover types using remote sensing and GIS techniques'. International Journal of Science and Engineering Investigations, 7(72), pp.11-18.
5. Patro, R.N., Subudhi, S., Biswal, P.K. and Dell’acqua, F. (2021). 'A review of unsupervised band selection techniques: Land cover classification for hyperspectral earth observation data'. IEEE Geoscience and Remote Sensing Magazine, 9(3), pp.72-111.
6. Camps-Valls, G., Martino, L., Svendsen, D.H., Campos-Taberner, M., Muñoz-Marí, J., Laparra, V., Luengo, D. and García-Haro, F.J. (2018). 'Physics-aware Gaussian processes in remote sensing'. Applied Soft Computing, 68, pp.69-82.
7. Chen, W., Tsamados, M., Willatt, R., Brockley, D., Deisenroth, M., De Rijke-Thomas, C., Francis, A., Hirata, L., Johnson, T., Lawrence, I. and Landy, J. (2024). 'Co-located OLCI optical imagery and SAR altimetry from Sentinel-3 for enhanced surface classification in sea ice (No. EGU24-9175)'. Copernicus Meetings.
8. Wang, Q., Shi, W., Li, Z. and Atkinson, P.M. (2016). 'Fusion of Sentinel-2 images'. Remote sensing of environment, 187, pp.241-252.
