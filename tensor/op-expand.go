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
	if ope.axis == 0 {
		tensor.adjustShape(Shape{ope.copies, ope.a.shape.Y})
	} else {
		tensor.adjustShape(Shape{ope.a.shape.X, ope.a.shape.Y * ope.copies})
	}
	C.expand(ope.a._tensor, C.int(ope.axis), C.int(ope.copies), tensor._tensor)
}

func (ope *opExpand) backward(tensor *Tensor) {
	ope.a.SetGradient(mat.Sum(tensor.GradientToMat32(), 0).Data())
}

func Expand(a *Tensor, axis, copies int) *Tensor {
	result := Variable(copies, a.shape.Y)
	if axis == 1 {
		result.Reshape(a.shape.X, a.shape.Y * copies)
	}
	result.op = &opExpand{a, axis, copies}
	return result
}
