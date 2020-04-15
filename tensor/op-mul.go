package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR}  -lmul
//#include <mul.h>
import "C"

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationMul = "opMul"

type opMul struct {
	a, b *Tensor
}

func (om *opMul) name() string {
	return operationMul
}

func (om *opMul) dependencies() []*Tensor {
	return []*Tensor{om.a, om.b}
}

func (om *opMul) forward(tensor *Tensor) {
	C.mul(om.a._tensor, om.b._tensor, tensor._tensor)
	//tensor.ConvertToRegularData()
	//tensor.SetData(mat.Mul(om.a.mat, om.b.mat).Data())
}


func (om *opMul) backward(tensor *Tensor) {
	if om.a.isGradientEnabled {
		om.a.grad = mat.Mul(tensor.grad, om.b.mat)
	}

	if om.b.isGradientEnabled {
		om.b.grad = mat.Mul(tensor.grad, om.a.mat)
	}
}

func Mul(a, b *Tensor) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opMul{a, b}
	return result
}
