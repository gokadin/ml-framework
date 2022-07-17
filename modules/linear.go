package modules

import (
	"fmt"
	"ml-framework/mat"
	"ml-framework/tensor"
)

type LinearModule struct {
	Type          string         `json:"type"`
	Weights       *tensor.Tensor `json:"weights"`
	Bias          *tensor.Tensor `json:"bias"`
	UnitCount     int            `json:"unit_count"`
	IsInitialized bool           `json:"is_initialized"`
}

func Linear(unitCount int) *LinearModule {
	return &LinearModule{
		Type:      "LinearModule",
		UnitCount: unitCount,
	}
}

func (d *LinearModule) Build(input *tensor.Tensor) *tensor.Tensor {
	if !d.IsInitialized {
		d.Weights = tensor.FromMat32(mat.Initialize(mat.InitXavier, mat.Dim(input.Shape().D[1], d.UnitCount))).
			SetName(fmt.Sprintf("LinearModule layer (%d) weights", d.UnitCount))
		d.Bias = tensor.Zeros(1, d.UnitCount).SetName(fmt.Sprintf("LinearModule layer (%d) biases", d.UnitCount))
		d.IsInitialized = true
	}

	return tensor.Linear(input, d.Weights, d.Bias)
}

func (d *LinearModule) GetParameters() []*tensor.Tensor {
	return []*tensor.Tensor{d.Weights, d.Bias}
}

func (d *LinearModule) Copy() Module {
	module := Linear(d.UnitCount)
	module.IsInitialized = d.IsInitialized
	module.Weights = d.Weights.Copy()
	module.Bias = d.Bias.Copy()
	return module
}

func (d *LinearModule) GetType() string {
	return d.Type
}
