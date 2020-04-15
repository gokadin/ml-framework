package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR}  -lrelu
//#include <relu.h>
import "C"

import (
	"github.com/gokadin/ml-framework/mat"
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
	C.relu(opw.a._tensor, tensor._tensor)
	//tensor.ConvertToRegularData()
	//tensor.mat = mat.Apply(tensor.mat, func(value float32) float32 {
	//	if value > 0 {
	//		return value
	//	}
	//	return 0
	//})
}

func (opw *opRelu) backward(tensor *Tensor) {
	d := mat.Apply(tensor.mat, func(value float32) float32 {
		if value > 0 {
			return 1
		}
		return 0
	})
	opw.a.grad = mat.Mul(tensor.grad, d)
}

func Relu(a *Tensor) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opRelu{a}
	return result
}
