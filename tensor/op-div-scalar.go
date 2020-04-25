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
	tensor.adjustShape(opw.a.shape)
	tensor.SetData(mat.DivScalar(opw.a.ToMat32f(), opw.scalar).Data())
}

func (opw *opDivScalar) backward(tensor *Tensor) {
	multiplier := 1.0 / opw.scalar
	opw.a.SetGradient(mat.MulScalar(tensor.GradientToMat32(), multiplier).Data())
}

func DivScalar(a *Tensor, scalar float32) *Tensor {
	result := Variable(a.shape.X, a.shape.Y)
	result.op = &opDivScalar{a, scalar}
	return result
}
