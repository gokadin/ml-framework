package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationRelu = "opRelu"

type opRelu struct {
	a *Tensor
}

func (opw *opRelu) name() string {
	return operationRelu
}

func (opw *opRelu) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opRelu) forward(tensor *Tensor) {
	tensor.mat = mat.Apply(opw.a.mat, func(value float32) float32 {
		if value > 0 {
			return value
		}
		return 0
	})
}

func (opw *opRelu) backward(tensor *Tensor) {
	d := mat.Apply(tensor.mat, func(value float32) float32 {
		if value > 0 {
			return 1
		}
		return 0
	})
	opw.a.grad = mat.Mul(tensor.grad, d)
}

func Relu(a *Tensor) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opRelu{a}
	return result
}
