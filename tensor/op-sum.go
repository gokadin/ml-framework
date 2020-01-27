package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"log"
)

const operationSum = "opSum"

type opSum struct {
	a *Tensor
	axis int
}

func (ops *opSum) name() string {
	return operationSum
}

func (ops *opSum) dependencies() []*Tensor {
	return []*Tensor{ops.a}
}

func (ops *opSum) forward(tensor *Tensor) {
	switch ops.axis {
	case 0:
		result := make([][]float64, 1)
		result[0] = make([]float64, len(ops.a.mat[0]))
		for i := range ops.a.mat {
			for j := range ops.a.mat[i] {
				result[0][j] += ops.a.mat[i][j]
			}
		}
		tensor.mat = result
		break
	case 1:
		result := make([][]float64, len(ops.a.mat))
		for i := range ops.a.mat {
			result[i] = make([]float64, 1)
			for j := range ops.a.mat[i] {
				result[i][0] += ops.a.mat[i][j]
			}
		}
		tensor.mat = result
		break
	default:
		log.Fatal("sum only supports axis 0 and 1")
	}
}

func (ops *opSum) backward(tensor *Tensor) {
	ops.a.grad = mat.Expand(tensor.grad, 0, len(ops.a.mat))
}

func Sum(a *Tensor, axis int) *Tensor {
	shapeX := 1
	shapeY := len(a.mat[0])
	if axis == 1 {
		shapeX = len(a.mat)
		shapeY = 1
	}
	result := Variable(shapeX, shapeY)
	result.op = &opSum{a, axis}
	return result
}
