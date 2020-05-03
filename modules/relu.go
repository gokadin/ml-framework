package modules

import (
	"github.com/gokadin/ml-framework/tensor"
)

type relu struct {}

func Relu() *relu {
	return &relu{}
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

