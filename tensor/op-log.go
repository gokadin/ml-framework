package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"math"
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
	for i := range tensor.mat {
		for j := range tensor.mat[i] {
			tensor.mat[i][j] = math.Log(opw.a.mat[i][j])
		}
	}
}

func (opw *opLog) backward(tensor *Tensor) {
	opw.a.grad = mat.DivScalarBy(tensor.grad, 1)
}

func Log(a *Tensor) *Tensor {
	result := Variable(len(a.mat), len(a.mat[0]))
	result.op = &opLog{a}
	return result
}
