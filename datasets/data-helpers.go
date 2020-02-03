package datasets

import (
	"github.com/gokadin/ml-framework/mat"
	"log"
)

func bytesToMat(bytes []byte, numOfSamples, headerOffset int) *mat.Mat32f {
	if (len(bytes) - headerOffset) % numOfSamples != 0 {
		log.Fatal("Could not transform raw data into usable Dataset")
	}

	sampleLength := (len(bytes) - headerOffset) / numOfSamples
	data := make([]float32, numOfSamples * sampleLength)
	for i := 0; i < numOfSamples; i++ {
		for j := 0; j < sampleLength; j++ {
			data[i * numOfSamples + j] = float32(bytes[headerOffset + (i * sampleLength + j)])
		}
	}

	return mat.NewMat32f(mat.WithShape(numOfSamples, sampleLength), data)
}

func oneHotEncode(mat *mat.Mat32f, depth int) {
	if mat.Shape().X == 0 || mat.Shape().Y != 1 {
		log.Fatal("cannot one hot encode Dataset with more than one value per output")
	}

	for i := 0; i < mat.Shape().X; i++ {
		value := int(mat.At(i * mat.Shape().X))
		for j := 0; j < depth; j++ {
			if value == j {
				mat.Set(i * mat.Shape().X + j, 1)
			} else {
				mat.Set(i * mat.Shape().X + j, 0)
			}
		}
	}
}

func normalize(mat *mat.Mat32f, min, max float32) {
	mat.Apply(func(value float32) float32 {
		return (value - min) / (max - min)
	})
}