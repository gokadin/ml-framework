package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationDot = "opDot"

type opDot struct {
	a, b *Tensor
}

func (od *opDot) name() string {
	return operationDot
}

func (od *opDot) dependencies() []*Tensor {
	return []*Tensor{od.a, od.b}
}

func (od *opDot) forward(tensor *Tensor) {
	tensor.mat = mat.MatMulParallel(od.a.mat, od.b.mat)
}

func (od *opDot) backward(tensor *Tensor) {
	if od.a.isGradientEnabled {
		od.a.grad = mat.MatMulParallel(tensor.grad, mat.Transpose(od.b.mat))
	}
	if od.b.isGradientEnabled {
		od.b.grad = mat.Transpose(mat.MatMulParallel(mat.Transpose(tensor.grad), od.a.mat))
	}
}

func Dot(a, b *Tensor) *Tensor {
	result := Variable(mat.WithShape(a.mat.Shape().X, b.mat.Shape().Y))
	result.op = &opDot{a, b}
	return result
}
