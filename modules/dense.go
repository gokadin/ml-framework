package modules

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/tensor"
)

const defaultWeightInitializer = initializerTypeNormalized
const defaultBiasInitializer = initializerTypeZeros

type dense struct {
	activation string
	weightInitializer string
	biasInitializer string
	isOutputLayer bool
	weights *tensor.Tensor
	bias *tensor.Tensor
	unitCount int
}

func Dense(unitCount int, activation string) *dense {
	return &dense{
		activation: 			activation,
		weightInitializer:		defaultWeightInitializer,
		biasInitializer: 		defaultBiasInitializer,
		isOutputLayer:          false,
		unitCount:              unitCount,
	}
}

func (d *dense) Forward(input *tensor.Tensor) *tensor.Tensor {
	if d.weights == nil {
		d.initialize(input.Shape().Y)
	}

	return activate(tensor.Add(tensor.Dot(input, d.weights), tensor.Expand(d.bias, 0, input.Shape().X)), d.activation)
}

func (d *dense) GetParameters() []*tensor.Tensor {
	return []*tensor.Tensor{d.weights, d.bias}
}

func (d *dense) initialize(inputSize int) {
	d.weights = initializeParameter(mat.WithShape(inputSize, d.unitCount), d.weightInitializer).SetName("dense layer weights")
	d.bias = initializeParameter(mat.WithShape(1, d.unitCount), d.biasInitializer).SetName("dense layer biases")
}