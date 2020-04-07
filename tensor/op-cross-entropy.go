package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR}  -lcrossentropy
//#include <cross-entropy.h>
import "C"

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationCrossEntropy = "opCrossEntropy"

type opCrossEntropy struct {
	a *Tensor
	target *Tensor
}

func (opw *opCrossEntropy) name() string {
	return operationCrossEntropy
}

func (opw *opCrossEntropy) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opCrossEntropy) forward(tensor *Tensor) {
	//tensor.SetData(mat.DivScalar(mat.Sum(mat.Neg(mat.Log(mat.Sum(mat.Mul(opw.target.mat, opw.a.mat), 1))), 0), float32(opw.a.mat.Shape().X)).Data())
	C.cross_entropy(opw.a._tensor, opw.target._tensor, tensor._tensor)
	tensor.SetData(tensor.TempData())
}

func (opw *opCrossEntropy) backward(tensor *Tensor) {
	expandedGrad := mat.Expand(mat.Expand(tensor.grad, 1, opw.a.mat.Shape().Y), 0, opw.a.mat.Shape().X)
	opw.a.grad = mat.Mul(expandedGrad, mat.Sub(opw.a.mat, opw.target.mat))
}

func CrossEntropy(pred, target *Tensor) *Tensor {
	result := Variable(mat.WithShape(3, 1))
	result.op = &opCrossEntropy{pred, target}
	return result
}
