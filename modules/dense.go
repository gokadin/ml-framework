package modules

import "github.com/gokadin/ml-framework/tensor"

const defaultWeightInitializer = initializerTypeNormalized
const defaultBiasInitializer = initializerTypeZeros

type dense struct {
	activation string // make callable
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

func (d *dense) Forward(input *tensor.Tensor) *tensor.Tensor { // can optimize the expand here
	if d.weights == nil {
		d.initialize(len(input.Data()[0]))
	}

	return activate(tensor.Add(tensor.Dot(input, d.weights), tensor.Expand(d.bias, 0, len(input.Data()))), d.activation)
}

func (d *dense) Backward() {

}

func (d *dense) initialize(inputSize int) {
	d.weights = initializeParameter(inputSize, d.unitCount, d.weightInitializer)
	d.bias = initializeParameter(1, d.unitCount, d.biasInitializer)
}