# Face-Detection
Deep Learning | Computer Vision
# Part B: Face Detection using Haarcascade
## Objective:
To detect and extract facial regions from raw profile images using classical computer vision (OpenCV's Haarcascade).
## Dataset:
- Folder containing profile images of individuals
- Format: .jpg files
- Size: 1000 images (approx)
- Source: Provided by an educational institution.

Note: This dataset cannot be shared publicly due to institutional policy. Please use your own images for face detection testing.
##  Methodology:
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
  
# Part C: Face Recognition using Embeddings
📌 Objective:
To recognize faces by computing embeddings with a pretrained VGGFace model and classifying them using SVM.

📁 Dataset:
PINS dataset containing aligned face images of 100 celebrities.

Format: Folder structure with subfolders per person.

Size: ~10,770 images

Source: Provided as part of coursework; reuse is restricted.

⚠️ Cannot share dataset publicly. You may use any labeled face dataset like LFW or your own.

🔧 Methodology:
Load images and extract metadata

Preprocess and normalize face inputs

Compute face embeddings using pretrained VGGFace model

Use PCA for dimensionality reduction

Train SVM classifier on embeddings

Predict identity of test face images

Evaluate cosine similarity between embeddings
