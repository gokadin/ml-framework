package tensor

import (
	"log"
	"math"
	"math/rand"
)

const (
	initRandom     = "initRandom"
	initNormalized = "initNormalized"
	initXavier     = "initXavier"
)

func initializeParameter(initializerType string, shapeArray ...int) *Tensor {
	switch initializerType {
	case initRandom:
		return initializeParameterRandom(shapeArray...)
	case initNormalized:
		return initializeParameterNormalized(shapeArray...)
	case initXavier:
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
	limit := float32(math.Sqrt(6.0 / float64(t.Shape().X + t.Shape().Y)))
	data := make([]float32, t.Shape().X * t.Shape().Y)
	for i := range data {
		data[i] = -limit + rand.Float32()  * (limit + limit)
	}
	return t.SetData(data)
}
