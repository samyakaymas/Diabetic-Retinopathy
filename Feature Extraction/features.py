import math
import numpy as np
import imutils
import cv2
import time
import progressbar
from os import listdir
from time import sleep
from sklearn import svm,preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from imutils import contours
from collections import Counter

class feature_extraction:
	def __init__(self):
		# Initialising the features and labels.
		self.Microaneuryms_Count = np.zeros((159,1))
		self.Microaneuryms_Area = np.zeros((159,1))
		self.HardExudates_Count = np.zeros((159,1))
		self.HardExudates_Area = np.zeros((159,1))
		self.HardExudate_Density = np.zeros((159,1))
		self.SoftExudates_Count = np.zeros((159,1))
		self.SoftExudates_Area = np.zeros((159,1))
		self.Hemorrhages_Count = np.zeros((159,1))
		self.Hemorrhages_Area = np.zeros((159,1))
		self.BloodVessel_Density = np.zeros((159,1))
		self.StandardDeviation_Red = np.zeros((159,1))
		self.StandardDeviation_Green = np.zeros((159,1))
		self.StandardDeviation_Blue = np.zeros((159,1))
		self.Entropy_Green = np.zeros((159,1))
		self.y = np.ones((89))
		self.X = np.zeros((159,8))
		self.path = None
	def Extract_Microaneuryms(self):
		# Extracting Microaneuryms for 89 images.
		imagesList = listdir(self.path)
		print("Extracting Microaneurysms:")
		bar = progressbar.ProgressBar(maxval=89,widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		i=1
		bar.start()
		for image in imagesList:
			self.Microaneuryms(image)
			bar.update(i)
			i+=1
		bar.finish()

	def Microaneuryms(self,image_name):
		# Extracting Count and Area of microaneuryms from already microaneuryms-extracted image.
		number = 0
		area = 0
		image = cv2.imread(self.path+image_name,0)

		# Using Contours, finding number of circular spots(microaneuryms).
		# Filtering the spots based on area.
		im2, contours, hierarchy = cv2.findContours(image.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			if cv2.contourArea(cnt) < 2000:
				number+=1
				area+=cv2.contourArea(cnt)
		i = int(image_name[5:8]) - 1

		# Updating the Count and Area
		self.Microaneuryms_Count[i] = number
		self.Microaneuryms_Area[i] = area

	def Extract_HardExudate_Density(self):
		# Extracting Hard Exudates Density for 89 images.
		imagesList = listdir(self.path)
		print("Extracting Hard Exudates:")
		bar = progressbar.ProgressBar(maxval=89,widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		i=1
		bar.start()
		for image in imagesList:
			self.HardExudates(image)
			bar.update(i)
			i+=1
		bar.finish()

	def HardExudates(self,image_name):
		# Extracting Count and Area of Hard Exudates from already hard-exudates-extracted image.
		number = 0
		area = 0
		image = cv2.imread(self.path+image_name,0)

		# Using contours, filtering the spots whose area is less than 1000000.
		im2, contours, hierarchy = cv2.findContours(image.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			if cv2.contourArea(cnt) < 1000000:
				number+=1
				area+=cv2.contourArea(cnt)
		i = int(image_name[5:8]) - 1

		# Updating hard exudates count,area and density.
		self.HardExudates_Count[i] = number
		self.HardExudates_Area[i] = area
		self.HardExudate_Density[i] = area / ( image.shape[0] * image.shape[1] )

	def Extract_SoftExudates(self):
		# Extracting Soft Exudates for 89 images.
		imagesList = listdir(self.path)
		for image in imagesList:
			self.SoftExudates(image)

	def SoftExudates(self,image_name):
		# Extracting Count and Area of Soft Exudates from already soft-exudates-extracted image.
		number = 0
		area = 0
		image = cv2.imread(self.path+image_name,0)
		im2, contours, hierarchy = cv2.findContours(image.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			if cv2.contourArea(cnt) > 24:
				number+=1
				area+=cv2.contourArea(cnt)
		i = int(image_name[5:8]) - 1
		self.SoftExudates_Count[i] = number
		self.SoftExudates_Area[i] = area

	def Extract_Hemorrhages(self):
		# Extracting Hamorrhages for 89 images.
		imagesList = listdir(self.path)
		for image in imagesList:
			self.Hemorrhages(image)

	def Hemorrhages(self,image_name):
		# Extracting Count and Area of Hemorrhages from already hemorrhages-extracted image.
		number = 0
		area = 0
		image = cv2.imread(self.path+image_name,0)
		im2, contours, hierarchy = cv2.findContours(image.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			number+=1
			area+=cv2.contourArea(cnt)
		i = int(image_name[5:8]) - 1
		self.Hemorrhages_Count[i] = number
		self.Hemorrhages_Area[i] = area

	def Extract_BloodVessels(self):
		# Extracting Blood Vessels for 89 images.
		imagesList = listdir(self.path)
		print("Extracting Blood Vessels:")
		bar = progressbar.ProgressBar(maxval=89,widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		i=1
		bar.start()
		for image in imagesList:
			self.BloodVessel(image)
			bar.update(i)
			i+=1
		bar.finish()

	def BloodVessel(self,image_name):
		# Extracting density of blood vessels for an image.
		image = cv2.imread(self.path+image_name)

		# Spliting the BGR channel of the image
		blue, green, red = cv2.split(image)

		# Normalizing and blurring the green channel with window size 11x21.
		dst = np.zeros(shape=(11,21))
		normalized = cv2.normalize(green,dst,0,255,cv2. NORM_MINMAX)
		blur = cv2.medianBlur(normalized,3)

		# Applying Contrast Limited Adaptive Histogram Equalization 3 times.
		clahe = cv2.createCLAHE(clipLimit = 1.0, tileGridSize=(12,12))
		adhe = clahe.apply(blur)
		adhe = clahe.apply(adhe)
		adhe = clahe.apply(adhe)

		# Extracting binary image of blood vessel.
		bv1 = self.BloodVesselImage(adhe)

		# Calculating Area of blood vessels (count of white pixels) and Dividing it by total pixel to get density.
		total_white_pixels = np.sum(bv1) / 255
		density = total_white_pixels / (bv1.shape[0]*bv1.shape[1])
		i = int(image_name[5:8]) - 1
		self.BloodVessel_Density[i] = density

	def BloodVesselImage(self,green):
		# Extracting binary blood vessels image from the green component of the image.
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12,12))
		contrast_enhanced_green_fundus = clahe.apply(green)

		# Applying a series of morphological open-close operations to reduce the noise from image.
		# Applying opening will reduce the noise of size according to window size.
		# Applying closing after opening will ensure blood vessels aren't removed by the next opening operation.
		# Increasing window size will help reducing small noise and relative larger noise also.
		r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)), iterations = 1)
		R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
		r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)), iterations = 1)
		R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)), iterations = 1)
		r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(29,29)), iterations = 1)
		R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(29,29)), iterations = 1)

		# Subtracting the original image from morphological operated image further reduces exudates and other noise.
		f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
		f5 = clahe.apply(f4)

		# Applying thresholding to get blood vessels along with some impurities most likely to be microaneuryms.
		# Thats also how microaneuryms are extracted i.e. subtracting only blood vessels from this impure image.
		ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)

		# Using contours and masking, filtering image and removing spots whose area is less than 200 i.e. microaneuryms.
		mask = np.ones(f5.shape[:2], dtype="uint8") * 255
		im2, contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			if cv2.contourArea(cnt) <= 200:
				cv2.drawContours(mask, [cnt], -1, 0, -1)
		im = cv2.bitwise_and(f5, f5, mask=mask)
		ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
		newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
		fundus_eroded = cv2.bitwise_not(newfin)

		# Using contours, filtering image and removing circular structures(they are not blood vessels).
		xmask = np.ones(green.shape[:2], dtype="uint8") * 255
		ymask = np.ones(green.shape[:2], dtype="uint8") * 255
		x1, xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in xcontours:
			shape = "unidentified"
			peri = cv2.arcLength(cnt, True)
			approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
			if len(approx) > 4 and cv2.contourArea(cnt) <= 2050 and cv2.contourArea(cnt) >= 20:
				shape = "circle"
			else:
				shape = "veins"
			if(shape=="circle"):
				cv2.drawContours(xmask, [cnt], -1, 0, -1)
			else:
				cv2.drawContours(ymask, [cnt], -1, 0, -1)

		# Finally dilating the image.
		finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=ymask)
		blood_vessels = cv2.bitwise_not(finimage)
		dilated = cv2.erode(blood_vessels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)
		blood_vessels_1 = cv2.bitwise_not(dilated)
		return finimage

	def Extract_StandardDeviation_Entropy(self):
		# Extracting Standard Deviation for all BGR components and Entropy of green component for 89 images.
		imagesList = listdir(self.path)
		print("Extracting Standard Deviation and Entropy:")
		bar = progressbar.ProgressBar(maxval=89,widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		i=1
		bar.start()
		for image in imagesList:
			self.StandardDeviation_Entropy(image)
			bar.update(i)
			i+=1
		bar.finish()

	def StandardDeviation_Entropy(self,image_name):
		# Standard Deviation for red blue green component.
		image = cv2.imread(self.path+image_name)

		# Splitting all 3 channels.
		blue, green, red = cv2.split(image)
		a,r = cv2.meanStdDev(red)
		a,b = cv2.meanStdDev(blue)
		a,g = cv2.meanStdDev(green)
		i = int(image_name[5:8]) - 1
		self.StandardDeviation_Red[i] = r
		self.StandardDeviation_Blue[i] = b
		self.StandardDeviation_Green[i] = g
		self.Entropy_Green[i] = self.Entropy(green)

	def Entropy(self,green):
		# Calculating Entropy for green component of an image.
		# Preparing count of all the pixels from 0 to 255
		count = Counter(green.flatten())
		en = 0
		total = green.shape[0] * green.shape[1]

		# Entropy is the sum of -probability*log(probabilty) of occurance of pixels.
		# Probabilty = Occurance / Total Pixels
		for i,f in count.items():
			if i!=0:
				en -= f/total * math.log(f/total)
		return en

	def GaussianNoise(self,data,mean,sigma):
		row,col= data.shape
		data2 = data.copy()
		gauss = np.random.normal(mean,sigma**(0.5),(row,col))
		gauss = gauss.reshape((row,col))
		data2 = data2 + gauss
		return data2

	def Extract(self):
		# Extracting every feature using this method.
		# Differnt paths for the directory of images.
		path1 = "redsmalldots/"
		path2 = "hardexudates/"
		path3 = "softexudates/"
		path4 = "hemorrhages/"
		path5 = "images/"
		self.path = path1
		self.Extract_Microaneuryms()
		self.path = path2
		self.Extract_HardExudate_Density()
		self.path = path3
		self.Extract_SoftExudates()
		self.path = path4
		self.Extract_Hemorrhages()
		self.path = path5
		self.Extract_BloodVessels()
		self.Extract_StandardDeviation_Entropy()

		# We have total 89 images out of which 5 are non diabetic.
		# So Copying 70 images of non-diabetic eye to counter class imbalancing.
		self.BloodVessel_Density[89:] = self.BloodVessel_Density[48]
		self.StandardDeviation_Red[89:] = self.StandardDeviation_Red[48]
		self.StandardDeviation_Blue[89:] = self.StandardDeviation_Blue[48]
		self.StandardDeviation_Green[89:] = self.StandardDeviation_Green[48]
		self.Entropy_Green[89:] = self.StandardDeviation_Green[48]

		# The non-diabetic samples.
		y1 = [48,56,59,61,71]

		# y is currently 1 for all samplse.
		# extending 70 0's in y and labelling non-diabetic samples 0.
		self.y = np.append(self.y,np.zeros(70))
		for i in y1:
			self.y[i] = 0

		# Updating the final Feature Array X of size 159x8.
		for i in range(159):
			self.X[i][0] = self.Microaneuryms_Count[i]
			self.X[i][1] = self.Microaneuryms_Area[i]
			self.X[i][2] = self.BloodVessel_Density[i]
			self.X[i][3] = self.StandardDeviation_Red[i]
			self.X[i][4] = self.StandardDeviation_Blue[i]
			self.X[i][5] = self.StandardDeviation_Green[i]
			self.X[i][6] = self.HardExudate_Density[i]
			self.X[i][7] = self.Entropy_Green[i]

		# Saving the features and labels to a csv files to be used in MATLAB for training.
		np.savetxt("../Datasets/WITHOUT_NOISE (1).csv",np.append(self.X , np.reshape(self.y,(159,1)),axis=1), delimiter = ",")
		self.X = self.GaussianNoise(self.X,1000,100)
		np.savetxt("../Datasets/WITH_NOISE (1).csv",np.append(self.X , np.reshape(self.y,(159,1)),axis=1), delimiter = ",")


if __name__ == "__main__":
	# Creating object of feature_extraction
	F = feature_extraction()
	start = time.time()
	# Extracting all the features.
	F.Extract()
	print("Time Taken for Extraction :",int(time.time()-start),'seconds.')
	print("\n\n")
	# Training and Testing the extracted features on sklearn SVM.
	print("Testing the features using sklearn svm classifier.")
	X,y = F.X,F.y
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
	clf = svm.SVC(gamma = 'scale')
	start = time.time()
	clf.fit(X_train,y_train)
	print("Time Taken for Training :",round(time.time()-start,2),'seconds.')
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	accuracy_train = accuracy_score(y_train,pred_train) * 100
	accuracy_test = accuracy_score(y_test,pred_test) * 100
	print("Accuracy on Training Set =",round(accuracy_train,8),'%')
	print("Accuracy on 20% Test Set =",round(accuracy_test,8),'%')
