package modules

import (
	"ml-framework/tensor"
)

type tanh struct {
	Type string `json:"type"`
}

func Tanh() *tanh {
	return &tanh{
		Type: "tanh",
	}
}

func (d *tanh) Build(input *tensor.Tensor) *tensor.Tensor {
	return tensor.Tanh(input)
}

func (d *tanh) GetParameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

func (d *tanh) Copy() Module {
	return Tanh()
}

func (d *tanh) GetType() string {
	return d.Type
}
