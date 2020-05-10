package tensor

import (
	"log"
	"math"
	"math/rand"
)

const (
	InitRandom        = "InitRandom"
	InitNormalized    = "InitNormalized"
	InitXavier        = "InitXavier"
)

func initializeParameter(initializerType string, shapeArray ...int) *Tensor {
	switch initializerType {
	case InitRandom:
		return initializeParameterRandom(shapeArray...)
	case InitNormalized:
		return initializeParameterNormalized(shapeArray...)
	case InitXavier:
		return initializeParameterXavier(shapeArray...)
	}

	log.Fatalf("parameter initializer of type %s is unknown", initializerType)
	return nil
}

func initializeParameterRandom(shapeArray ...int) *Tensor {
	t := OfShape(shapeArray...)
	data := make([]float32, t.Shape().Size())
	for i := range data {
		data[i] = rand.Float32()
	}
	return t.SetData(data)
}

func initializeParameterNormalized(shapeArray ...int) *Tensor {
	t := OfShape(shapeArray...)
	data := make([]float32, t.Shape().Size())
	for i := range data {
		data[i] = rand.Float32() / float32(math.Sqrt(float64(t.Shape().X)))
	}
	return t.SetData(data)
}

func initializeParameterXavier(shapeArray ...int) *Tensor {
	t := OfShape(shapeArray...)
	limit := float32(math.Sqrt(6.0 / float64(t.Shape().X+t.Shape().Y)))
	data := make([]float32, t.Shape().X*t.Shape().Y)
	for i := range data {
		data[i] = -limit + rand.Float32()*(limit+limit)
	}
	return t.SetData(data)
}
