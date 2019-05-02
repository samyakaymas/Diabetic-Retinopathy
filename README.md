# Diabetic-Retinopathy
This code presents an improved diabetic retinopathy detection scheme by using variants of Support Vector Machines(SVMs).
As Twin Support Vector Machine(TWSVM) are ruling the classifiers in various manners, we have considered their use in our work.
Also, to add noise, we have considered another variant of TWSVM, pin-TWSVM. 
In this work, we have used several features namely;
number and area of  Microaneuryms
the density of Hard Exudates and Blood Vessels
Standard Deviation and Entropy
Classification of DR using these features has been done by Linear SVM, Twin SVM with Hinge Loss, and Twin SVM with Pinball Loss.
The conclusion is drawn for both the noisy and noise-free dataset.
From this work, we have concluded that pin-TWSVM is the best choice for diabetic retinopathy detection both in terms of accuracy and robustness to noise.
As the previous works show that the SVM is working well for this problem, our work provides better result than the existing works using SVM.
It should be noted here that it is for the first time that twin SVM and its variants are used for the problem of diabetic retionopathy. 
This choice of classifier also helped in reducing the time consumption for classification of diabetic and non-diabetic corresponding to this dataset.
