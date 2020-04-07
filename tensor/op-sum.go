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
	originalShape mat.Shape
}

func (ops *opSum) name() string {
	return operationSum
}

func (ops *opSum) dependencies() []*Tensor {
	return []*Tensor{ops.a}
}

func (ops *opSum) forward(tensor *Tensor) {
	C.sum(ops.a._tensor, C.int(ops.axis), tensor._tensor)
	tensor.SetData(tensor.TempData())
}

func (ops *opSum) backward(tensor *Tensor) {
	if ops.axis == 0 {
		ops.a.grad = mat.Expand(tensor.grad, 0, ops.originalShape.X)
	} else if ops.axis == 1 {
		ops.a.grad = mat.Expand(tensor.grad, 1, ops.originalShape.Y)
	}
}

func Sum(a *Tensor, axis int) *Tensor {
	var result *Tensor
	if axis == 0 {
		result = Variable(mat.WithShape(1, a.Shape().Y))
	} else if axis == 1 {
		result = Variable(mat.WithShape(a.Shape().X, 1))
	} else {
		log.Fatal("axis unknown: " + string(axis))
	}
	result.op = &opSum{a, axis, a.mat.Shape()}
	return result
}
