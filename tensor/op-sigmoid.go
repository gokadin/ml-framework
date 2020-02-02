package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"math"
)

const operationSigmoid = "opSigmoid"

type opSigmoid struct {
	a *Tensor
}

func (opw *opSigmoid) name() string {
	return operationSigmoid
}

func (opw *opSigmoid) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opSigmoid) forward(tensor *Tensor) {
	for i := range tensor.mat {
		for j := range tensor.mat[i] {
			tensor.mat[i][j] = 1 / (math.Exp(-opw.a.mat[i][j]) + 1)
		}
	}
}

func (opw *opSigmoid) backward(tensor *Tensor) {
	opw.a.grad = mat.Mul(tensor.grad, mat.Mul(tensor.mat, mat.SubFromScalar(tensor.mat, 1)))
}

func Sigmoid(a *Tensor) *Tensor {
	result := Variable(len(a.mat), len(a.mat[0]))
	result.op = &opSigmoid{a}
	return result
}
