package modules

import (
	"github.com/gokadin/ml-framework/tensor"
)

type leakyRelu struct {}

func LeakyRelu() *leakyRelu {
	return &leakyRelu{}
}

func (d *leakyRelu) Build(input *tensor.Tensor) *tensor.Tensor {
	return tensor.LeakyRelu(input)
}

func (d *leakyRelu) GetParameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

func (d *leakyRelu) Copy() Module {
	return LeakyRelu()
}
