package mat

import (
	"log"
	"math"
	"math/rand"
)

const (
	InitRandom     = "InitRandom"
	InitNormalized = "InitNormalized"
	InitXavier     = "InitXavier"
)

func Initialize(initializerType string, dim ShapeN) *Mat32f {
	switch initializerType {
	case InitRandom:
		return initializeParameterRandom(dim)
	case InitNormalized:
		return initializeParameterNormalized(dim)
	case InitXavier:
		return initializeParameterXavier(dim)
	}

	log.Fatalf("parameter initializer of type %s is unknown", initializerType)
	return nil
}

func initializeParameterRandom(dim ShapeN) *Mat32f {
	data := make([]float32, dim.Size())
	for i := range data {
		data[i] = rand.Float32()
	}
	return NewMat32f(WithShape(dim.D[0], dim.D[1]), data)
}

func initializeParameterNormalized(dim ShapeN) *Mat32f {
	data := make([]float32, dim.Size())
	for i := range data {
		data[i] = rand.Float32() / float32(math.Sqrt(float64(dim.D[0])))
	}
	return NewMat32f(WithShape(dim.D[0], dim.D[1]), data)
}

func initializeParameterXavier(dim ShapeN) *Mat32f {
	limit := float32(math.Sqrt(6.0 / float64(dim.Size())))
	data := make([]float32, dim.Size())
	for i := range data {
		data[i] = -limit + rand.Float32()*(limit+limit)
	}
	return NewMat32f(WithShape(dim.D[0], dim.D[1]), data)
}
