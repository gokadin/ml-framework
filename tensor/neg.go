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
	tensor.SetData(mat.Neg(opn.a.ToMat32f()).Data())
}

func (opn *opNeg) backward(tensor *Tensor) {
	opn.a.SetGradient(tensor.GradientToFloat32())
}

func Neg(a *Tensor) *Tensor {
	result := OfShape(a.Shape().X, a.Shape().Y)
	result.op = &opNeg{a}
	return result
}
