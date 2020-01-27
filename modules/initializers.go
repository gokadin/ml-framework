package modules

import (
	"github.com/gokadin/ml-framework/tensor"
	"log"
	"math"
	"math/rand"
)

const (
	initializerTypeZeros = "initializerTypeZeros"
	initializerTypeRandom = "initializerTypeRandom"
	initializerTypeNormalized = "initializerTypeNormalized"
)

func initializeParameter(x, y int, initializerType string) *tensor.Tensor {
	switch initializerType {
	case initializerTypeZeros:
		return initializeParameterZeros(x, y)
	case initializerTypeRandom:
		return initializeParameterRandom(x, y)
	case initializerTypeNormalized:
		return initializeParameterNormalized(x, y)
	}

	log.Fatalf("parameter initializer of type %s is unknown", initializerType)
	return nil
}

func initializeParameterZeros(x, y int) *tensor.Tensor {
	mat := make([][]float64, x)
	for i := range mat {
		mat[i] = make([]float64, y)
		for j := range mat[i] {
			mat[i][j] = 0
		}
	}
	return tensor.Constant(mat)
}

func initializeParameterRandom(x, y int) *tensor.Tensor {
	mat := make([][]float64, x)
	for i := range mat {
		mat[i] = make([]float64, y)
		for j := range mat[i] {
			mat[i][j] = rand.NormFloat64()
		}
	}
	return tensor.Constant(mat)
}

func initializeParameterNormalized(x, y int) *tensor.Tensor {
	mat := make([][]float64, x)
	for i := range mat {
		mat[i] = make([]float64, y)
		for j := range mat[i] {
			mat[i][j] = rand.NormFloat64() / math.Sqrt(float64(x))
		}
	}
	return tensor.Constant(mat)
}
