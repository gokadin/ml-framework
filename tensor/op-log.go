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
	tensor.SetData(mat.Log(opw.a.mat).Data())
}

func (opw *opLog) backward(tensor *Tensor) {
	opw.a.grad = mat.DivScalarBy(tensor.grad, 1)
}

func Log(a *Tensor) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opLog{a}
	return result
}
