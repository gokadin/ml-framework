package modules

import (
	"fmt"
	"github.com/gokadin/ml-framework/tensor"
)

type linear struct {
	weights       *tensor.Tensor
	bias          *tensor.Tensor
	unitCount     int
	isInitialized bool
}

func Linear(unitCount int) *linear {
	return &linear{
		unitCount: unitCount,
	}
}

func (d *linear) Build(input *tensor.Tensor) *tensor.Tensor {
	if !d.isInitialized {
		d.weights = tensor.From(tensor.InitXavier, input.Shape().Y, d.unitCount).SetName(fmt.Sprintf("linear layer (%d) weights", d.unitCount))
		d.bias = tensor.Zeros(1, d.unitCount).SetName(fmt.Sprintf("linear layer (%d) biases", d.unitCount))
	}

	return tensor.Linear(input, d.weights, d.bias)
}

func (d *linear) GetParameters() []*tensor.Tensor {
	return []*tensor.Tensor{d.weights, d.bias}
}

func (d *linear) Copy() Module {
	module := Linear(d.unitCount)
	module.isInitialized = d.isInitialized
	module.weights = d.weights.Copy()
	module.bias = d.bias.Copy()
	return module
}
