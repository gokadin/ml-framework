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
	tensor.mat = mat.Apply(opw.a.mat, func(value float32) float32 {
		return float32(1 / (math.Exp(-float64(value)) + 1))
	})
}

func (opw *opSigmoid) backward(tensor *Tensor) {
	opw.a.grad = mat.Mul(tensor.grad, mat.Mul(tensor.mat, mat.SubFromScalar(tensor.mat, 1)))
}

func Sigmoid(a *Tensor) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opSigmoid{a}
	return result
}
