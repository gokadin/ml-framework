package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationDivScalar = "opDivScalar"

type opDivScalar struct {
	a *Tensor
	scalar float32
}

func (opw *opDivScalar) name() string {
	return operationDivScalar
}

func (opw *opDivScalar) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opDivScalar) forward(tensor *Tensor) {
	tensor.SetData(mat.DivScalar(opw.a.mat, opw.scalar).Data())
}

func (opw *opDivScalar) backward(tensor *Tensor) {
	multiplier := 1.0 / opw.scalar
	opw.a.grad = mat.MulScalar(tensor.grad, multiplier)
}

func DivScalar(a *Tensor, scalar float32) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opDivScalar{a, scalar}
	return result
}
