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
}

func (ope *opExpand) backward(tensor *Tensor) {
	ope.a.SetGradient(mat.Sum(tensor.GradientToMat32(), 0).Data())
}

func Expand(a *Tensor, axis, copies int) *Tensor {
	result := OfShape(copies, a.Shape().Y)
	if axis == 1 {
		result.Reshape(a.Shape().X, a.Shape().Y * copies)
	}
	result.op = &opExpand{a, axis, copies}
	return result
}
