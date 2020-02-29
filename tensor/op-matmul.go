package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationMatmul = "opMatmul"

type opMatmul struct {
	a, b *Tensor
}

func (omm *opMatmul) name() string {
	return operationMatmul
}

func (omm *opMatmul) dependencies() []*Tensor {
	return []*Tensor{omm.a, omm.b}
}

func (omm *opMatmul) forward(tensor *Tensor) {
	tensor.mat = mat.MatMulParallel(omm.a.mat, omm.b.mat)
}

func (omm *opMatmul) backward(tensor *Tensor) {
	if omm.a.isGradientEnabled {
		omm.a.grad = mat.MatMulParallel(tensor.grad, mat.Transpose(omm.b.mat))
	}
	if omm.b.isGradientEnabled {
		omm.b.grad = mat.Transpose(mat.MatMulParallel(mat.Transpose(tensor.grad), omm.a.mat))
	}
}

func Matmul(a, b *Tensor) *Tensor {
	result := Variable(mat.WithShape(a.mat.Shape().X, b.mat.Shape().Y))
	result.op = &opMatmul{a, b}
	return result
}