package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationPow = "opPow"

type opPow struct {
	a *Tensor
	power float32
}

func (opw *opPow) name() string {
	return operationPow
}

func (opw *opPow) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opPow) forward(tensor *Tensor) {
	tensor.SetData(mat.Pow(opw.a.mat, float64(opw.power)).Data())
}

func (opw *opPow) backward(tensor *Tensor) {
	if opw.power == 2 {
		opw.a.grad = mat.Mul(tensor.grad, mat.MulScalar(opw.a.mat, 2))
		return
	}
	opw.a.grad = mat.Mul(tensor.grad, mat.MulScalar(mat.Pow(opw.a.mat, float64(opw.power) - 1), opw.power))
}

func Pow(a *Tensor, power float32) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opPow{a, power}
	return result
}
