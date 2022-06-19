package modules

import (
	"ml-framework/tensor"
)

type relu struct {
	Type string `json:"type"`
}

func Relu() *relu {
	return &relu{
		Type: "relu",
	}
}

func (d *relu) Build(input *tensor.Tensor) *tensor.Tensor {
	return tensor.Relu(input)
}

func (d *relu) GetParameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

func (d *relu) Copy() Module {
	return Relu()
}

func (d *relu) GetType() string {
	return d.Type
}
