# Predict-group-emotions-using-individual-facial-expressions
Predict group emotions using individual facial expressions

Due to size of this code & associated dataset, we have to put many artifacts to .gitignore. However, that can be procured and create as follows;

For this project to run at your side, you need to do following:
1. Create dataset folder at root level
2. Within dataset folder create FacialExpression and dump FER dataset with all 7 emotions folders & their associated sample image files
3. Create Scraped-Dataset folder and dump image dataset
4. If you change above folder name, then you need to update them in code files: ModelTrain.py & Predict.py
5. Similarly you need to put yolomodels weights for face detection in folder in YoloModels
6. You need to create Model folder
7. Within Models folder you need to create FineTunedModelForCeleb
8. First we need to pip install requirements.txt to install required binaries / libraries
9. Then first we need to run ModelTrain.py, this will create pre-trained model weights on emotions based on FER dataset and save model weight in Modles/Facial_expression_recognition_weights.pth
10. Then we need to run Predict.py to further fine tuned earlier pre-trained model on FER dataset on scraped / celebrity dataset.
11. We use yolo v8 face detection model and then fine tuned our fine tuned model to identify individual facial expression and then cummulate it to identify group emotions for image.
12. Similarly we can do the same for frame to identify facial expression
13. Use that frame routine to identify facial expression on video
14. Video can be video input or live feed / camera.
15. Use App.py with streamlit app to run app to get user input for supervised & unsupervise learning & processing
16. Beside this we have FeatureEngineering.py to do HOB, ldb & facial landmark from input images
17. We use DataSetPrep.py to extract features from images dataset
18. We use TrainAndTunedSupervised.py to do supervised traiing and do cnn custom training, train & evaluate cnn, train & evaluate svm & train & evaluate random forest
19. We use TrainAndTunedSupervised.py to plot model performance, plot hyperparameter tuning, plot model metric comparision, plot confusion matrix, plot precision recall curve, plot roc curve, plot cnn training curve
20. We do hyper paramter tunning through TrainAndTunedSupervisor.py
21. We used KMeansClustering.py & PCAVisualization.py for unsupervised learning
22. We use autoencoder.py for unuserpivse learning for encoding of images.

    
