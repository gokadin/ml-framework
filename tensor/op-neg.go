package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationNeg = "opNeg"

type opNeg struct {
	a *Tensor
}

func (opn *opNeg) name() string {
	return operationNeg
}

func (opn *opNeg) dependencies() []*Tensor {
	return []*Tensor{opn.a}
}

func (opn *opNeg) forward(tensor *Tensor) {
	tensor.SetData(mat.Neg(opn.a.mat).Data())
}

func (opn *opNeg) backward(tensor *Tensor) {
	opn.a.grad = tensor.grad
}

func Neg(a *Tensor) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opNeg{a}
	return result
}
