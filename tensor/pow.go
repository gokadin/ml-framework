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
	tensor.SetData(mat.Pow(opw.a.ToMat32f(), float64(opw.power)).Data())
}

func (opw *opPow) backward(tensor *Tensor) {
	if opw.power == 2 {
		opw.a.SetGradient(mat.Mul(tensor.GradientToMat32(), mat.MulScalar(opw.a.ToMat32f(), 2)).Data())
		return
	}
	opw.a.SetGradient(mat.Mul(tensor.GradientToMat32(), mat.MulScalar(mat.Pow(opw.a.ToMat32f(), float64(opw.power) - 1), opw.power)).Data())
}

func Pow(a *Tensor, power float32) *Tensor {
	result := OfShape(a.Shape().X, a.Shape().Y)
	result.op = &opPow{a, power}
	return result
}
