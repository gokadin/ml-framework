package datasets

import (
	"log"
)

func bytesToMat(bytes []byte, numOfSamples, headerOffset int) [][]float64 {
	if (len(bytes) - headerOffset) % numOfSamples != 0 {
		log.Fatal("Could not transform raw data into usable dataset")
	}

	sampleLength := (len(bytes) - headerOffset) / numOfSamples
	mat := make([][]float64, numOfSamples)
	for i := 0; i < numOfSamples; i++ {
		sampleMat := make([]float64, sampleLength)
		for j := 0; j < sampleLength; j++ {
			sampleMat[j] = float64(bytes[headerOffset + (i * sampleLength + j)])
		}
		mat[i] = sampleMat
	}

	return mat
}

func oneHotEncode(data [][]float64, depth int) {
	if len(data) == 0 || len(data[0]) != 1 {
		log.Fatal("cannot one hot encode dataset with more than one value per output")
	}

	for i, output := range data {
		value := int(output[0])
		encoded := make([]float64, depth)
		for j := 0; j < depth; j++ {
			if value == j {
				encoded[j] = 1.0
			} else {
				encoded[j] = 0.0
			}
		}
		data[i] = encoded
	}
}