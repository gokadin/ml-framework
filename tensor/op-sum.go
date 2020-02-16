package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationSum = "opSum"

type opSum struct {
	a *Tensor
	axis int
	originalShape mat.Shape
}

func (ops *opSum) name() string {
	return operationSum
}

func (ops *opSum) dependencies() []*Tensor {
	return []*Tensor{ops.a}
}

func (ops *opSum) forward(tensor *Tensor) {
	tensor.mat = mat.Sum(ops.a.mat, ops.axis)
	//switch ops.axis {
	//case 0:
	//	result := make([][]float64, 1)
	//	result[0] = make([]float64, len(ops.a.mat[0]))
	//	for i := range ops.a.mat {
	//		for j := range ops.a.mat[i] {
	//			result[0][j] += ops.a.mat[i][j]
	//		}
	//	}
	//	tensor.mat = result
	//	break
	//case 1:
	//	result := make([][]float64, len(ops.a.mat))
	//	for i := range ops.a.mat {
	//		result[i] = make([]float64, 1)
	//		for j := range ops.a.mat[i] {
	//			result[i][0] += ops.a.mat[i][j]
	//		}
	//	}
	//	tensor.mat = result
	//	break
	//default:
	//	log.Fatal("sum only supports axis 0 and 1")
	//}
}

func (ops *opSum) backward(tensor *Tensor) {
	if ops.axis == 0 {
		ops.a.grad = mat.Expand(tensor.grad, 0, ops.originalShape.X)
	} else if ops.axis == 1 {
		ops.a.grad = mat.Expand(tensor.grad, 1, ops.originalShape.Y)
	}
}

func Sum(a *Tensor, axis int) *Tensor {
	result := Variable(mat.WithShape(1, a.Shape().Y))
	if axis == 1 {
		result.Reshape(mat.WithShape(a.Shape().X, 1))
	}
	result.op = &opSum{a, axis, a.mat.Shape()}
	return result
}
