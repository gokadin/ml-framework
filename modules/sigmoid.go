package modules

import (
	"github.com/gokadin/ml-framework/tensor"
)

type sigmoid struct {}

func Sigmoid() *sigmoid {
	return &sigmoid{}
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

