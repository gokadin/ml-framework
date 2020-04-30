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
}


func (om *opMul) backward(tensor *Tensor) {
	if om.a.isGradientEnabled {
		om.a.SetGradient(mat.Mul(tensor.GradientToMat32(), om.b.ToMat32f()).Data())
	}

	if om.b.isGradientEnabled {
		om.b.SetGradient(mat.Mul(tensor.GradientToMat32(), om.a.ToMat32f()).Data())
	}
}

func Mul(a, b *Tensor) *Tensor {
	result := Variable(a.Shape().X, a.Shape().Y)
	result.op = &opMul{a, b}
	return result
}
