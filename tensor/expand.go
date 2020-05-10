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

func (o *opExpand) name() string {
	return operationExpand
}

func (o *opExpand) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opExpand) forwardShape() Shape {
	return Shape{o.a.Shape().X, o.a.Shape().Y * o.copies}
}

// TODO
func (o *opExpand) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape, tensorShape}
}

func (o *opExpand) forward(tensor *Tensor) {
	C.expand(o.a._tensor, C.int(o.axis), C.int(o.copies), tensor._tensor)
}

func (o *opExpand) backward(tensor *Tensor) {
	o.a.SetGradient(mat.Sum(tensor.GradientToMat32(), 0).Data())
}

func Expand(a *Tensor, axis, copies int) *Tensor {
	result := OfShape(copies, a.Shape().Y)
	if axis == 1 {
		result.Reshape(a.Shape().X, a.Shape().Y * copies)
	}
	result.op = &opExpand{a, axis, copies}
	return result
}
