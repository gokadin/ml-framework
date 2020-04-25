package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationLog = "opLog"

type opLog struct {
	a *Tensor
}

func (opw *opLog) name() string {
	return operationLog
}

func (opw *opLog) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opLog) forward(tensor *Tensor) {
	tensor.adjustShape(opw.a.shape)
	tensor.SetData(mat.Log(opw.a.ToMat32f()).Data())
}

func (opw *opLog) backward(tensor *Tensor) {
	opw.a.SetGradient(mat.DivScalarBy(tensor.GradientToMat32(), 1).Data())
}

func Log(a *Tensor) *Tensor {
	result := Variable(a.shape.X, a.shape.Y)
	result.op = &opLog{a}
	return result
}
