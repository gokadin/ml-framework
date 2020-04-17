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

func initializeParameter(initializerType string, shapeArray ...int) *tensor.Tensor {
	switch initializerType {
	case initializerTypeZeros:
		return initializeParameterZeros(shapeArray...)
	case initializerTypeRandom:
		return initializeParameterRandom(shapeArray...)
	case initializerTypeNormalized:
		return initializeParameterNormalized(shapeArray...)
	case initializerTypeXavier:
		return initializeParameterXavier(shapeArray...)
	}

	log.Fatalf("parameter initializer of type %s is unknown", initializerType)
	return nil
}

func initializeParameterZeros(shapeArray ...int) *tensor.Tensor {
	t := tensor.Variable(shapeArray...)
	return t.SetData(mat.Zeros32f(t.Size()))
}

func initializeParameterRandom(shapeArray ...int) *tensor.Tensor {
	t := tensor.Variable(shapeArray...)
	data := make([]float32, t.Size())
	for i := range data {
		data[i] = rand.Float32()
	}
	return t.SetData(data)
}

func initializeParameterNormalized(shapeArray ...int) *tensor.Tensor {
	t := tensor.Variable(shapeArray...)
	data := make([]float32, t.Size())
	for i := range data {
		data[i] = rand.Float32() / float32(math.Sqrt(float64(t.Shape().X)))
	}
	return t.SetData(data)
}

func initializeParameterXavier(shapeArray ...int) *tensor.Tensor {
	t := tensor.Variable(shapeArray...)
	limit := float32(math.Sqrt(6.0 / float64(t.Shape().X + t.Shape().Y)))
	data := make([]float32, t.Shape().X * t.Shape().Y)
	for i := range data {
		data[i] = -limit + rand.Float32()  * (limit + limit)
	}
	return t.SetData(data)
}
