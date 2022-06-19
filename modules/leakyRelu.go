package modules

import (
	"ml-framework/tensor"
)

type leakyRelu struct {
	Type string `json:"type"`
}

func LeakyRelu() *leakyRelu {
	return &leakyRelu{
		Type: "leakyRelu",
	}
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

func (d *leakyRelu) GetType() string {
	return d.Type
}
