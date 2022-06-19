package modules

import (
	"ml-framework/tensor"
)

type sigmoid struct {
	Type string `json:"type"`
}

func Sigmoid() *sigmoid {
	return &sigmoid{
		Type: "sigmoid",
	}
}

func (d *sigmoid) Build(input *tensor.Tensor) *tensor.Tensor {
	return tensor.Sigmoid(input)
}

func (d *sigmoid) GetParameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

func (d *sigmoid) Copy() Module {
	return Sigmoid()
}

func (d *sigmoid) GetType() string {
	return d.Type
}
