package datasets

import (
	"log"
	"ml-framework/mat"
)

const mnistName = "mnist"
const baseUrl = "http://yann.lecun.com/exdb/mnist/"
const trainingSetImagesFilename = "train-images-idx3-ubyte.gz"
const trainingSetLabelsFilename = "train-labels-idx1-ubyte.gz"
const validationSetImagesFilename = "t10k-images-idx3-ubyte.gz"
const validationSetLabelsFilename = "t10k-labels-idx1-ubyte.gz"
const trainingSetCount = 60000
const validationSetCount = 10000
const trainingImagesHeaderOffset = 16
const trainingLabelsHeaderOffset = 8
const validationImagesHeaderOffset = 16
const validationLabelsHeaderOffset = 8

func loadMNIST() *Dataset {
	log.Print("Getting MNIST Dataset...")

	trainX, trainY, valX, valY := getData()
	trainY = oneHotEncode(trainY, 10)
	valY = oneHotEncode(valY, 10)
	normalize(trainX, 0, 255, -1, 1)
	normalize(valX, 0, 255, -1, 1)

	dataset := NewDataset()
	dataset.SetName(mnistName)
	dataset.AddData(TrainingSetX, trainX)
	dataset.AddData(TrainingSetY, trainY)
	dataset.AddData(ValidationSetX, valX)
	dataset.AddData(ValidationSetY, valY)
	return dataset
}

func getData() (*mat.Mat32f, *mat.Mat32f, *mat.Mat32f, *mat.Mat32f) {
	trainingImagesBytes := unzip(downloadFile(baseUrl, trainingSetImagesFilename))
	trainingLabelsBytes := unzip(downloadFile(baseUrl, trainingSetLabelsFilename))
	validationImagesBytes := unzip(downloadFile(baseUrl, validationSetImagesFilename))
	validationLabelsBytes := unzip(downloadFile(baseUrl, validationSetLabelsFilename))

	return bytesToMat(trainingImagesBytes, trainingSetCount, trainingImagesHeaderOffset),
		bytesToMat(trainingLabelsBytes, trainingSetCount, trainingLabelsHeaderOffset),
		bytesToMat(validationImagesBytes, validationSetCount, validationImagesHeaderOffset),
		bytesToMat(validationLabelsBytes, validationSetCount, validationLabelsHeaderOffset)
}
