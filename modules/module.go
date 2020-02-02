package modules

import (
	"github.com/gokadin/ml-framework/tensor"
)

type Module interface {
	Forward(input *tensor.Tensor) *tensor.Tensor
	GetParameters() []*tensor.Tensor
}
