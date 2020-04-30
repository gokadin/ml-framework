package tensor

//#include <matmul.h>
//#include <linear.h>
import "C"

import "github.com/gokadin/ml-framework/mat"

const operationLinear = "opLinear"

type opLinear struct {
	a, x, b *Tensor
}

func (oa *opLinear) name() string {
	return operationLinear
}

func (oa *opLinear) dependencies() []*Tensor {
	return []*Tensor{oa.a, oa.x, oa.b}
}

func (oa *opLinear) forward(tensor *Tensor) {
	//C.linear(oa.a._tensor, oa.x._tensor, oa.b._tensor, tensor._tensor)
	//tensor.SetData(mat.Add(mat.MatMulParallel(oa.a.ToMat32f(), oa.x.ToMat32f()), mat.Expand(oa.b.ToMat32f(), 0, oa.a.Shape().X)).Data())
}

func (oa *opLinear) backward(tensor *Tensor) {
	oa.b.SetGradient(mat.Sum(mat.NewMat32f(mat.WithShape(tensor.Shape().X, tensor.Shape().Y), tensor.GradientToFloat32()), 0).Data())
	C.gpu_matmul_backward(tensor._tensor, oa.a._tensor, oa.x._tensor)
	//if oa.a.isGradientEnabled {
		//omm.b.ConvertToRegularData()
		//oa.a.SetGradient(mat.MatMulParallel(tensor.GradientToMat32(), mat.Transpose(oa.x.ToMat32f())).Data())
	//}
	//if oa.x.isGradientEnabled {
		//omm.a.ConvertToRegularData()
		//oa.x.SetGradient(mat.Transpose(mat.MatMulParallel(mat.Transpose(tensor.GradientToMat32()), oa.a.ToMat32f())).Data())
	//}
}

func Linear(a, x, b *Tensor) *Tensor {
	result := OfShape(a.Shape().X, b.Shape().Y)
	result.op = &opLinear{a, x, b}
	//return Add(Matmul(a, x), Expand(b, 0, a.Shape().X))
	result._tensor.op = C.alloc_linear(a._tensor, x._tensor, b._tensor)
	return result
}
