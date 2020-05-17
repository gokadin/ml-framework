package modules

import (
	"github.com/gokadin/ml-framework/tensor"
)

type tanh struct {}

func Tanh() *tanh {
	return &tanh{}
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

