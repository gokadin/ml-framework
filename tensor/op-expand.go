package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationExpand = "opExpand"

type opExpand struct {
	a *Tensor
	axis int
	copies int
}

func (ope *opExpand) name() string {
	return operationExpand
}

func (ope *opExpand) dependencies() []*Tensor {
	return []*Tensor{ope.a}
}

func (ope *opExpand) forward(tensor *Tensor) {
	tensor.mat = mat.Expand(ope.a.mat, ope.axis, ope.copies)
	//switch ope.axis {
	//case 0:
	//	result := make([][]float64, ope.copies)
	//	for i := range result {
	//		result[i] = make([]float64, len(ope.a.mat[0]))
	//		for j := range ope.a.mat[0] {
	//			result[i][j] = ope.a.mat[0][j]
	//		}
	//	}
	//	tensor.mat = result
	//	break
	//case 1:
	//	result := make([][]float64, len(ope.a.mat))
	//	for i := range ope.a.mat {
	//		result[i] = make([]float64, len(ope.a.mat[i]) * ope.copies)
	//		copyCounter := 0
	//		for j := 0; j < ope.copies; j++ {
	//			for k := range ope.a.mat[i] {
	//				result[i][copyCounter] = ope.a.mat[i][k]
	//				copyCounter++
	//			}
	//		}
	//	}
	//	tensor.mat = result
	//	break
	//default:
	//	log.Fatal("sum only supports axis 0 and 1")
	//}
}

func (ope *opExpand) backward(tensor *Tensor) {
	ope.a.grad = mat.Sum(tensor.grad, 0)
}

func Expand(a *Tensor, axis, copies int) *Tensor {
	result := Variable(mat.WithShape(copies, a.mat.Shape().Y))
	if axis == 1 {
		result.mat.Reshape(mat.WithShape(a.mat.Shape().X, a.mat.Shape().Y * copies))
	}
	result.op = &opExpand{a, axis, copies}
	return result
}
