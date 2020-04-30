package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR}  -lsum
//#include <sum.h>
import "C"
import (
	"github.com/gokadin/ml-framework/mat"
	"log"
)

const operationSum = "opSum"

type opSum struct {
	a *Tensor
	axis int
	originalShape Shape
}

func (ops *opSum) name() string {
	return operationSum
}

func (ops *opSum) dependencies() []*Tensor {
	return []*Tensor{ops.a}
}

func (ops *opSum) forward(tensor *Tensor) {
	C.sum(ops.a._tensor, C.int(ops.axis), tensor._tensor)
}

func (ops *opSum) backward(tensor *Tensor) {
	if ops.axis == 0 {
		ops.a.SetGradient(mat.Expand(tensor.GradientToMat32(), 0, ops.originalShape.X).Data())
	} else if ops.axis == 1 {
		ops.a.SetGradient(mat.Expand(tensor.GradientToMat32(), 1, ops.originalShape.Y).Data())
	}
}

func Sum(a *Tensor, axis int) *Tensor {
	var result *Tensor
	if axis == 0 {
		result = Variable(1, a.Shape().Y)
	} else if axis == 1 {
		result = Variable(a.Shape().X, 1)
	} else {
		log.Fatal("axis unknown: " + string(axis))
	}
	result.op = &opSum{a, axis, a.Shape()}
	return result
}
