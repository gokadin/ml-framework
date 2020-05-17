package modules

import (
	"github.com/gokadin/ml-framework/tensor"
)

type logSoftmax struct {}

func LogSoftmax() *logSoftmax {
	return &logSoftmax{}
}

func (d *logSoftmax) Build(input *tensor.Tensor) *tensor.Tensor {
	return tensor.LogSoftmax(input)
}

func (d *logSoftmax) GetParameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

func (d *logSoftmax) Copy() Module {
	return LogSoftmax()
}

