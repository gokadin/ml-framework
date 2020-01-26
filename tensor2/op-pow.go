package tensor2

import (
	"github.com/gokadin/ml-framework/mat"
	"math"
)

const operationPow = "opPow"

type opPow struct {
	a *Tensor
	power float64
}

func (opw *opPow) name() string {
	return operationPow
}

func (opw *opPow) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opPow) forward(mat [][]float64) {
	for i := range mat {
		for j := range mat[i] {
			mat[i][j] = math.Pow(opw.a.mat[i][j], opw.power)
		}
	}
}

func (opw *opPow) backward(grad [][]float64) {
	if opw.power == 2 {
		opw.a.grad = mat.Mul(grad, mat.MulScalar(opw.a.mat, 2))
	}
	opw.a.grad = mat.Mul(grad, mat.MulScalar(mat.Pow(opw.a.mat, opw.power - 1), opw.power))
}

func Pow(a *Tensor, power float64) *Tensor {
	result := Variable(len(a.mat), len(a.mat[0]))
	result.op = &opPow{a, power}
	return result
}
