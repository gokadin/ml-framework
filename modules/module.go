package modules

import (
	"github.com/gokadin/ml-framework/tensor"
)

type Module interface {
	Build(input *tensor.Tensor) *tensor.Tensor
	GetParameters() []*tensor.Tensor
	Copy() Module
}
