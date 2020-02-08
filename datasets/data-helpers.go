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
			data[i * sampleLength + j] = float32(bytes[headerOffset + (i * sampleLength + j)])
		}
	}

	return mat.NewMat32f(mat.WithShape(numOfSamples, sampleLength), data)
}

func oneHotEncode(m *mat.Mat32f, depth int) *mat.Mat32f {
	if m.Shape().X == 0 || m.Shape().Y != 1 {
		log.Fatal("cannot one hot encode Dataset with more than one value per output")
	}

	result := mat.Expand(m, 1, depth)
	for i := 0; i < result.Shape().X; i++ {
		value := int(result.At(i * depth))
		for j := 0; j < depth; j++ {
			if value == j {
				result.Set(i * depth + j, 1)
			} else {
				result.Set(i * depth + j, 0)
			}
		}
	}
	return result
}

func normalize(mat *mat.Mat32f, min, max float32) {
	mat.Apply(func(value float32) float32 {
		return (value - min) / (max - min)
	})
}