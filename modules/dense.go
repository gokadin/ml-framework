package modules

import (
	"fmt"
	"github.com/gokadin/ml-framework/tensor"
)

const defaultWeightInitializer = initializerTypeXavier
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

func (d *dense) Type() string {
	return "dense"
}

func (d *dense) Initialize(inputSize int) {
	if d.weights != nil {
		return
	}

	weightsTensor := initializeParameter(d.weightInitializer, inputSize, d.unitCount)
	biasTensor := initializeParameter(d.biasInitializer, 1, d.unitCount)
	d.InitializeWith(weightsTensor, biasTensor)
}

func (d *dense) InitializeWith(weights, biases *tensor.Tensor) {
	if d.weights != nil {
		return
	}

	d.weights = weights.SetName(fmt.Sprintf("dense layer (%d) weights", d.unitCount))
	d.bias = biases.SetName(fmt.Sprintf("dense layer (%d) biases", d.unitCount))
}

func (d *dense) Forward(input *tensor.Tensor) *tensor.Tensor {
	return activate(tensor.Linear(input, d.weights, d.bias), d.activation)
}

func (d *dense) GetParameters() []*tensor.Tensor {
	return []*tensor.Tensor{d.weights, d.bias}
}

func (d *dense) GetActivation() string {
	return d.activation
}
