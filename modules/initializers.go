package modules

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/tensor"
	"log"
	"math"
	"math/rand"
)

const (
	initializerTypeZeros = "initializerTypeZeros"
	initializerTypeRandom = "initializerTypeRandom"
	initializerTypeNormalized = "initializerTypeNormalized"
	initializerTypeXavier = "initializerTypeXavier"
)

func initializeParameter(shape mat.Shape, initializerType string) *tensor.Tensor {
	switch initializerType {
	case initializerTypeZeros:
		return initializeParameterZeros(shape)
	case initializerTypeRandom:
		return initializeParameterRandom(shape)
	case initializerTypeNormalized:
		return initializeParameterNormalized(shape)
	case initializerTypeXavier:
		return initializeParameterXavier(shape)
	}

	log.Fatalf("parameter initializer of type %s is unknown", initializerType)
	return nil
}

func initializeParameterZeros(shape mat.Shape) *tensor.Tensor {
	return tensor.Constant(mat.NewMat32fZeros(shape))
}

func initializeParameterRandom(shape mat.Shape) *tensor.Tensor {
	data := make([]float32, shape.X * shape.Y)
	for i := range data {
		data[i] = rand.Float32()
	}
	return tensor.Constant(mat.NewMat32f(shape, data))
}

func initializeParameterNormalized(shape mat.Shape) *tensor.Tensor {
	data := make([]float32, shape.X * shape.Y)
	for i := range data {
		data[i] = rand.Float32() / float32(math.Sqrt(float64(shape.X)))
	}
	return tensor.Constant(mat.NewMat32f(shape, data))
}

func initializeParameterXavier(shape mat.Shape) *tensor.Tensor {
	limit := float32(math.Sqrt(6.0 / float64(shape.X + shape.Y)))
	data := make([]float32, shape.X * shape.Y)
	for i := range data {
		data[i] = -limit + rand.Float32()  * (limit + limit)
	}
	return tensor.Constant(mat.NewMat32f(shape, data))
}
