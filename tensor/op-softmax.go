package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationSoftmax = "opSoftmax"

type opSoftmax struct {
	a *Tensor
}

func (opw *opSoftmax) name() string {
	return operationSoftmax
}

func (opw *opSoftmax) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opSoftmax) forward(tensor *Tensor) {
	tensor.mat = mat.Softmax(opw.a.mat)
}

func (opw *opSoftmax) backward(tensor *Tensor) {
	opw.a.grad = tensor.grad
}

func Softmax(a *Tensor) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opSoftmax{a}
	return result
}
