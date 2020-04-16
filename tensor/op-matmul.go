package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR}  -lmatmul
//#include <matmul.h>
import "C"

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationMatmul = "opMatmul"

type opMatmul struct {
	a, b *Tensor
}

func (omm *opMatmul) name() string {
	return operationMatmul
}

func (omm *opMatmul) dependencies() []*Tensor {
	return []*Tensor{omm.a, omm.b}
}

func (omm *opMatmul) forward(tensor *Tensor) {
	C.matmul(omm.a._tensor, omm.b._tensor, tensor._tensor)
	//tensor.ConvertToRegularData()
	//tensor.SetData(mat.MatMulParallel(omm.a.mat, omm.b.mat).Data())
}

func (omm *opMatmul) backward(tensor *Tensor) {
	tensor.convertToNativeGrad()
	C.matmul_backward(tensor._tensor, omm.a._tensor, omm.b._tensor)

	if omm.a.isGradientEnabled {
		omm.a.ConvertToRegularGrad()
		//omm.b.ConvertToRegularData()
		//omm.a.grad = mat.MatMulParallel(tensor.grad, mat.Transpose(omm.b.mat))
	}
	if omm.b.isGradientEnabled {
		omm.b.ConvertToRegularGrad()
		//omm.a.ConvertToRegularData()
		//omm.b.grad = mat.Transpose(mat.MatMulParallel(mat.Transpose(tensor.grad), omm.a.mat))
	}
}

func Matmul(a, b *Tensor) *Tensor {
	result := Variable(mat.WithShape(a.mat.Shape().X, b.mat.Shape().Y))
	result.op = &opMatmul{a, b}
	return result
}
