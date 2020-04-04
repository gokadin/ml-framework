package modules

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/tensor"
)

type Module interface {
	Type() string
	Forward(input *tensor.Tensor) *tensor.Tensor
	GetParameters() []*tensor.Tensor
	GetActivation() string
	Initialize(inputSize int)
	InitializeWith(weights, biases *mat.Mat32f)
}
