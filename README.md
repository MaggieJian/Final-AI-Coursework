# Surface Classification of Urban Cities using IRIS and Sentinel-2 Data

![5577E711-A665-4CC6-842F-0398D135C962_1_201_a](https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/1c5660ad-f78e-41ba-97d7-e552c2e03e98)

The purpose of this project is to explore the effectiveness of IRIS (Intelligently Reinforced Image Segmentation) for surface classification using an urban setting. It creates semi-supervised classified imagery that aims to differentiate between green spaces and urban walkways/buildings/etc., in the context of the Gardens by the Bay area of the south-eastern country of Singapore, which will be used as training data to critically analyse the performance of IRIS under three comparable supervised machine learning models: Convolutional Neural Network (CNN), Random Forest (RF) and Vision Transformer (ViT). The model performances were evaluated using model selection and a cross-validation score. As an extension, an unsupervised model (K-Means Clustering) was explored without IRIS using the same Sentinel-2 data. Overall, this project can be a starting point to understand ways to improve surface classification ML platforms and approaches of usage, extending to a wider range of future applications.

Summary of the results: The ViT model produced a better cross-validation score and was favoured in the model selection compared to the CNN and Random Forest models

This repository provides access to the relevant code, ipynb files, figures and version histories for the project.

Google Drive containing relevant files: 
[https://drive.google.com/drive/folders/1WH1KuUP78N7FZJCxplrzYwh-vaavqLaT]

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
  
Copernicus is a database platform that provides free access to satellite data and imagery that is close to real time. It was developed by the ESA-Phil lab, is managed by the European Commision and is contributed to by the Euporean Space Agency (ESA) and the European Environmental Agency (EEA). Currently it retrieves data from a satellite constellation series mostly named the Sentinel satellites to monitor air and water quality and to observe land use change, climate change and nataural disasters.
Sentinel-2 is one of the satellites with 13 spectral bands, each band with a resolution of either 10, 20 or 60 meters per pixel: 
<img width="634" alt="Screenshot 2024-05-20 at 16 46 48" src="https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/08574267-df79-4154-8a16-54ac56b471b6">
Image credit: Sentinel Online, 2020
Band 2 = Blue, Band 3 = Green, Band 4 = Red

Each band has different offsets from each other:
<img width="422" alt="Screenshot 2024-05-20 at 16 47 58" src="https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/def5a5d6-05c9-42ec-ba52-1e212e2f7baa">

Image credit: Sentinel Online, 2020

INSERT FIGURE 1

IRIS is a recent annotation tool developed over the past 4 years developed by the ESA Phil-Lab, which aims to make surface classification of multispectral and multimodel imagery a quicker and easier process for users. It leverages the iterative and sequential machine learning technique of artificial intelligence known as 'gradient boosted decision trees'. Each decision tree identifies errors from the previous tree to make corrections via a sequential manner. It can run on Linux, Windows and Mac OS (Wheeler, 2024).

* Supervised learning: Vision Transformers (ViT), CNN and Random Forests

In this project, IRIS will be used to create training and testing data for a ViT model, a CNN model and a Random Forest model. 

Over time, more complex models have been developed. It is generally agreed that Random Forests are the most traditional of the three models, followed by the two deep learning models, CNN and ViT. ViT has only recently been used in image processing in addition to natural language processing (Uparker et al., 2023).

INSERT FIGURE 2

Steps of CNN model:

Convolutional Neural Networks, commonly known as CNNs, are a class of deep neural networks specially designed to process data with grid-like topology, such as images {cite}Goodfellow-et-al-2016,lecun2015deep. Originating from the visual cortex's biological processes, CNNs are revolutionising the way we understand and interpret visual data.

Why CNN for Image Data?

Traditional neural networks, when used for images, suffer from two main issues:
Too many parameters: For a simple 256x256 colored image, an input layer would have (256 * 256 * 3 = 196,608) neurons, leading to an enormous number of parameters even in the first hidden layer.
Loss of spatial information: Flattening an image into a vector for traditional neural networks can lose the spatial hierarchies and patterns in the image, which are often crucial for understanding and interpreting visual data.
CNNs address both issues by introducing convolutions.

Key Components of CNN

Convolutional Layer {cite}lecun2015deep: This is the core building block of a CNN. It slides a filter (smaller in size than the input data) over the input data (like an image) to produce a feature map or convolved feature. The primary purpose of a convolution is to extract features from the input data.
Pooling Layer: Pooling layers are used to reduce the dimensions of the feature maps, thereby reducing the number of parameters and computation in the network. The most common type of pooling is max pooling.
Fully Connected Layer: After several convolutional and pooling layers, the final classification is done using one or more fully connected layers. Neurons in a fully connected layer have connections to all activations in the previous layer, as seen in regular neural networks.
Activation Functions: Non-linearity is introduced into the CNN using activation functions. The Rectified Linear Unit (ReLU) is the most commonly used activation function in CNNs.

How CNNs Learn Spatial Hierarchies

CNNs learn spatial hierarchies automatically. The initial layers might learn to detect edges, the next layers learn to detect shapes by combining edges, further layers might detect more complex structures. This ability to learn spatial hierarchies from raw data gives CNNs their power. It allows them to detect complex objects in images by combining simpler features from the earlier layers.

Advantages of CNNs

Parameter Sharing: A feature detector (filter) that's useful in one part of the image can be useful in another part of the image {cite}krizhevsky2012imagenet.
Sparsity of Connections: In each layer, each output value depends only on a small number of input values, making the computation more efficient.

Steps of Random Forests model:

Random Forest is a notable and significant part of machine learning and is commonly used for classification. It can also be used for regression, but its application in classification is more prevalent. Decision Trees are the core components of a Random Forest, so let's delve into the concepts of Decision Trees {cite}breiman2001random,quinlan1986induction.

1. Ensemble Learning

EL methods employ multiple learning algorithms to achieve better predictive performance than any individual learning algorithm alone {cite}dietterich2000ensemble. The primary principle behind ensemble models is that several weak learners come together to form a strong learner.

2. Decision Trees

Decision trees are central to a Random Forest. They split data into subsets based on feature values, recursively producing a decision tree {cite}quinlan1986induction.

3. Bootstrap Aggregating (Bagging)

Random Forests leverage bagging, where multiple dataset subsets are created by drawing samples with replacement. A separate decision tree is built for each of these samples {cite}breiman1996bagging.

4. Feature Randomness

In conventional decision trees, the best feature is chosen to split data at every node. However, Random Forests introduce randomness by selecting a random set of features, then choosing the best split from this subset, ensuring a diverse ensemble of trees.

Advantages

Generalisation: By combining the predictions of multiple trees, Random Forests tend to generalize better and are less susceptible to overfitting on training data.
Parallel Processing: Each decision tree can be built independently, allowing for parallel processing which speeds up the algorithm considerably for large datasets.
Handling Missing Values: Random Forests can handle missing values and still produce reasonable predictions.
Importance Scoring: They provide an importance score for each feature, aiding in feature selection or interpretability.

Steps of ViT model

Theoretical Foundations
1. Tokenisation of Images
Instead of processing images using convolutions, ViTs divide the image into fixed-size patches, linearly embed them, and then process the resulting sequence of vectors (or tokens) using a transformer.{cite}dosovitskiy2020image

2. Position Embeddings
Since the original transformer doesn't have a notion of the relative positions of tokens, positional embeddings are added to the patch embeddings to retain the positional information.{cite}dosovitskiy2020image

3. Transformer Architecture
The core of ViT is the transformer architecture, which consists of multiple layers of multi-head self-attention mechanisms and feed-forward neural networks.{cite}dosovitskiy2020image

4. Classification Head
After processing through the transformer layers, the embedding of the first token (often referred to as the 'CLS' token) is used to classify the image.{cite}dosovitskiy2020image

Advantages of ViT

1. Model Transferability - ViTs pre-trained on large datasets can be fine-tuned on smaller datasets, achieving high performance even when the available labeled data is limited.

2. Scalability - ViTs are more data-hungry compared to CNNs. However, their performance continues to improve as the model size and the amount of data increase, often surpassing other architectures.

3. Flexibility - The transformer architecture isn't specialized for grid-like data (like images), making ViTs potentially more flexible for varied input data types.

Challenges

1. Computational Demand - ViTs can be computationally intensive, especially when dealing with large images or when the model has many layers.

2. Data Requirement - To achieve optimal performance, ViTs often require more training data compared to CNNs.

Implementation

The implmentation of Vision Transformer is much more complicated than CNN and Random Forest as there is no built-in functions or layers in the library.

* Method of evaluation

To compare the performance of IRIS with ViT, the comparison metrics will be typical of classification models. This includes a cross-validation score, accuracy, precision, recall, F1 score, support, macro average and weighted average:

![arhg](https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/077f0f41-b650-4d6d-b58a-886fdf12269d)
Image credit: https://en.wikipedia.org/wiki/Precision_and_recall

A confusion matrix will also help to evaluate the performance of IRIS.

# Methodology:

Step 1: Data Collection

Raw Sentinel-2 image data can be retrieved using Copernicus v1.10.1.

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

Work on a specific area of the image:
<img width="404" alt="Satellite view of mask area" src="https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/b96dd7a7-2ddc-40f2-a23c-5b81ffef91c6">

Troubleshooting tip: Ensure the shape of your image is reflected in the config.json file. If not, use .shape to print the shape then use your text editor to edit the config.json file.

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

Step 5: Model evaluation

The models will be assessed for overfitting and underfitting 

The metric to assess this will be a loss function, also known as a cost function, which quantifies how distant a model’s predictions are from the actual values. Minimising this value of the deviation from the actual results is preferred. Common loss functions include Mean Squared Error for regression tasks and Cross-Entropy for classification tasks.

The colab file to do this is named 'Comparing_supervised_learning_classification_models.ipynb'.

Extension: Unsupervised learning model

K-means clustering was used to visualise the Sentinel-2 satellite image without IRIS.

Use the colab file titled 'Unsupervised_learning.ipynb'.

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

After completing Steps 1-5, your should have a mask, test and training data, 3 saved models and a cross-validation score.

The mask created in this project is below:
![5577E711-A665-4CC6-842F-0398D135C962_1_201_a](https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/18d634b0-b701-4323-b8f7-b9c622f0b83c)

Using IRIS, green spaces were partially optically catagorised (~2% of the whole image was supervised training data, whereas the rest of the image was identified using IRIS' artificial intelligence system). This resulted in a semi-supervised classification. 
Note that during summer when this Sentinel-2 data was collected, increased foliage may have obscured the roads and created more shadows compared to in the winter.

After completing the extension, the following result was produced.

![kimage](https://github.com/MaggieJian/Final-AI-Coursework/assets/160494175/9aad64a2-d1d1-4440-bc3a-3f1295b17c10)

## Author contact and help request

Maggie Jian - maggie.jian.21@ucl.ac.uk

## License

This project is not licensed

## How to Cite

If you use this code or data in your work, please cite it as:
Jian, M. (2024). Final AI Coursework. GitHub Repository. https://github.com/MaggieJian/Week4

# Acknowledgments

This project is part of an assignment for the module GEOL0069 (2023/24) taught in thr Earth Sciences Department of University College London. Thank you to the module leaders and staff.

Inspiration and code snippets from:
[GEOL0069 Jupyter Notebook] (https://cpomucl.github.io/GEOL0069-AI4EO/intro.html)
[awesome-readme](https://github.com/matiassingers/awesome-readme)
[PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
[dbader](https://github.com/dbader/readme-template)

# References

Bai, Z., Wang, Y., Li, M., Sun, Y., Zhang, X., Wu, Y., Li, Y. and Li, D., 2023. Land Subsidence in the Singapore Coastal Area with Long Time Series of TerraSAR-X SAR Data. Remote Sensing, 15(9), pp.2415.

Ng, V. and Kim, H.M., 2021. Autonomous vehicles and smart cities: A case study of Singapore. In Smart cities for technological and social innovation. Academic Press, pp. 265-287.

Oyekola, M.A. and Adewuyi, G.K. (2018). 'Unsupervised classification in land cover types using remote sensing and GIS techniques'. International Journal of Science and Engineering Investigations, 7(72), pp.11-18.

Patro, R.N., Subudhi, S., Biswal, P.K. and Dell’acqua, F. (2021). 'A review of unsupervised band selection techniques: Land cover classification for hyperspectral earth observation data'. IEEE Geoscience and Remote Sensing Magazine, 9(3), pp.72-111.

Sentinel Online. (2020). MultiSpectral Instrument (MSI) Overview. European Space Agency.

Tsamados, M. and Chen, W. (2022). Introduction to Intelligently Reinforced Image Segmentation (IRIS) — GEOL0069 Guide Book. [online] Available at: https://cpomucl.github.io/GEOL0069-AI4EO/Chapter%201%3AIRIS.html [Accessed 12 May 2024].

Uparkar, O., Bharti, J., Pateriya, R.K., Gupta, R.K. and Sharma, A. (2023). Vision transformer outperforms deep convolutional neural network-based model in classifying X-ray images. Procedia Computer Science, 218, pp.2338-2349.

Wheeler, J. (2024). ESA-PhiLab/iris. [online] Available at: https://github.com/ESA-PhiLab/iris [Accessed 11 May 2024].

‌Yale Center for Business and the Environment. (2024). Marina Bay Sands. [online] Available at: https://cbey.yale.edu/research/marina-bay-sands#:~:text=Standing%20at%20water [Accessed May 2024].