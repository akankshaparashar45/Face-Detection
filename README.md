# Face-Detection
Deep Learning | Computer Vision
# 1.Installation
## 1.1 Install dependencies locally
  
  ```bash 
  pip install -r requirements.txt
  ```
 
Note: These libraries were installed and tested using Python 3.8.5.

# 2. The project is divided into three distinct parts, each addressing a specific computer vision task as described below.
## 2.1 Part A: Face Mask Detection
### 2.1.1 Objective:
To train a deep learning model to segment the facial region (mask) from movie scene images using U-Net and MobileNet architecture.
### 2.1.2 Dataset:
- Contains approximately 400 movie scene images with corresponding face annotations
- Format: NumPy array of (image, annotations) pairs
- Each entry includes:
   - X: RGB image array
   - Y: metadata with face coordinates and image dimensions
- Source: Provided by an educational institution

Note: The dataset is restricted and cannot be shared publicly. Please use your own image-mask pairs to replicate this task.
### 2.1.3 Methodology:
  (used in Face_Detection_Part A.ipynb)
- Data loading and preprocessing
- Normalization and shape unification
- Model building using U-Net + MobileNet architectures
- Train the model to learn face region segmentation
- Output: Predicted face masks for new input images
## 2.2 Part B: Face Detection using Haarcascade
### 2.2.1 Objective:
To detect and extract facial regions from raw profile images using classical computer vision (OpenCV’s Haarcascade), generate a metadata record for each detected face, and store the results in a Pandas DataFrame.
### 2.2.2 Dataset:
- Folder containing profile images of individuals
- Format: .jpg files
- Size: 1000 images (approx)
- Source: Provided by an educational institution

Note: This dataset cannot be shared publicly due to institutional policy. Please use your own images for face detection testing.
###  2.2.3 Methodology:
  (used in Face_Detection_Part B.ipynb)
- Load images using OpenCV
- Apply the pre-trained Haarcascade Frontal Face classifier
- Detect faces in each image and draw bounding boxes around detected regions
- Extract metadata for each face, including: 
   - Image name
   - x, y coordinates of the top-left corner
   - Width and height of the bounding box
   - Total number of faces detected in each image
- Store metadata in a Pandas DataFrame
- Export metadata to a .csv file
  
## 2.3 Part C: Face Recognition using Embeddings
### 2.3.1 Objective:
To recognize faces by computing embeddings with a pretrained VGG19 model and classifying them using SVM.
### 2.3.2 Dataset:
- Dataset containing aligned face images of 100 celebrities.
- Format: Folder structure with subfolders per person.
- Size: ~10,770 images
- Source: Provided as part of coursework; reuse is restricted.

Note: Cannot share dataset publicly. You may use any labeled face dataset.
### 2.3.3 Methodology:
  (used in Face_Detection_Part C.ipynb)
- Load images and extract metadata
- Preprocess and normalize face inputs
- Compute face embeddings using pretrained VGG19 model
- Evaluate cosine similarity between embeddings
- Use PCA for dimensionality reduction
- Train SVM classifier on embeddings
- Predict identity of test face images

