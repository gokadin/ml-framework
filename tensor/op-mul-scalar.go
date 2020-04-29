package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationMulScalar = "opMulScalar"

type opMulScalar struct {
	a *Tensor
	scalar float32
}

func (opw *opMulScalar) name() string {
	return operationMulScalar
}

func (opw *opMulScalar) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opMulScalar) forward(tensor *Tensor) {
	tensor.adjustShape(opw.a.shape)
	tensor.SetData(mat.MulScalar(opw.a.ToMat32f(), opw.scalar).Data())
}

func (opw *opMulScalar) backward(tensor *Tensor) {
	opw.a.SetGradient(mat.MulScalar(tensor.GradientToMat32(), opw.scalar).Data())
}

func MulScalar(a *Tensor, scalar float32) *Tensor {
	result := Variable(a.shape.X, a.shape.Y)
	result.op = &opMulScalar{a, scalar}
	return result
}
