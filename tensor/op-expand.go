package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR}  -lexpand
//#include <expand.h>
import "C"

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
	C.expand(ope.a._tensor, C.int(ope.axis), C.int(ope.copies), tensor._tensor)
	tensor.ConvertToRegularData()
	//tensor.SetData(mat.Expand(ope.a.mat, ope.axis, ope.copies).Data())
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
