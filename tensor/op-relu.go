package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"math"
)

const operationRelu = "opRelu"

type opRelu struct {
	a *Tensor
}

func (opw *opRelu) name() string {
	return operationRelu
}

func (opw *opRelu) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opRelu) forward(tensor *Tensor) {
	for i := range tensor.mat {
		for j := range tensor.mat[i] {
			tensor.mat[i][j] = math.Max(0, opw.a.mat[i][j])
		}
	}
}

func (opw *opRelu) backward(tensor *Tensor) {
	d := make([][]float64, len(tensor.mat))
	for i := range d {
		d[i] = make([]float64, len(tensor.mat[i]))
		for j := range d[i] {
			if tensor.mat[i][j] > 0 {
				d[i][j] = 1
			} else {
				d[i][j] = 0
			}
		}
	}
	opw.a.grad = mat.Mul(tensor.grad, d)
}

func Relu(a *Tensor) *Tensor {
	result := Variable(len(a.mat), len(a.mat[0]))
	result.op = &opRelu{a}
	return result
}
