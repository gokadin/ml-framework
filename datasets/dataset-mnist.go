package datasets

import (
	"log"
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

func loadMNIST() *dataset {
	log.Print("Getting MNIST dataset...")

	trainX, trainY, valX, valY := getData()
	oneHotEncode(trainY, 10)
	oneHotEncode(valY, 10)

	dataset := newDataset()
	dataset.setName(mnistName)
	dataset.addSet(trainingSetX, trainX)
	dataset.addSet(trainingSetY, trainY)
	dataset.addSet(validationSetX, valX)
	dataset.addSet(validationSetY, valY)
	return dataset
}

func getData() ([][]float64, [][]float64, [][]float64, [][]float64) {
	trainingImagesBytes := unzip(downloadFile(baseUrl, trainingSetImagesFilename))
	trainingLabelsBytes := unzip(downloadFile(baseUrl, trainingSetLabelsFilename))
	validationImagesBytes := unzip(downloadFile(baseUrl, validationSetImagesFilename))
	validationLabelsBytes := unzip(downloadFile(baseUrl, validationSetLabelsFilename))

	return bytesToMat(trainingImagesBytes, trainingSetCount, trainingImagesHeaderOffset),
		bytesToMat(trainingLabelsBytes, trainingSetCount, trainingLabelsHeaderOffset),
		bytesToMat(validationImagesBytes, validationSetCount, validationImagesHeaderOffset),
		bytesToMat(validationLabelsBytes, validationSetCount, validationLabelsHeaderOffset)
}