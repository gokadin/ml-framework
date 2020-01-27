package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationDivScalar = "opDivScalar"

type opDivScalar struct {
	a *Tensor
	scalar float64
}

func (opw *opDivScalar) name() string {
	return operationDivScalar
}

func (opw *opDivScalar) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opDivScalar) forward(tensor *Tensor) {
	for i := range opw.a.mat {
		for j := range opw.a.mat[i] {
			tensor.mat[i][j] = opw.a.mat[i][j] / opw.scalar
		}
	}
}

func (opw *opDivScalar) backward(tensor *Tensor) {
	opw.a.grad = mat.MulScalar(tensor.grad, 1 / opw.scalar)
}

func DivScalar(a *Tensor, scalar float64) *Tensor {
	result := Variable(len(a.mat), len(a.mat[0]))
	result.op = &opDivScalar{a, scalar}
	return result
}
