package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR} -llinear
//#include <linear.h>
//#include <matmul.h>
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
	tensor.adjustShape(Shape{oa.a.shape.X, oa.b.shape.Y})
	C.linear(oa.a._tensor, oa.x._tensor, oa.b._tensor, tensor._tensor)
	//tensor.SetData(mat.Add(mat.MatMulParallel(oa.a.ToMat32f(), oa.x.ToMat32f()), mat.Expand(oa.b.ToMat32f(), 0, oa.a.shape.X)).Data())
}

func (oa *opLinear) backward(tensor *Tensor) {
	oa.b.SetGradient(mat.Sum(mat.NewMat32f(mat.WithShape(tensor.shape.X, tensor.shape.Y), tensor.GradientToFloat32()), 0).Data())
	C.matmul_backward(tensor._tensor, oa.a._tensor, oa.x._tensor)
	if oa.a.isGradientEnabled {
		//omm.b.ConvertToRegularData()
		//oa.a.SetGradient(mat.MatMulParallel(tensor.GradientToMat32(), mat.Transpose(oa.x.ToMat32f())).Data())
	}
	if oa.x.isGradientEnabled {
		//omm.a.ConvertToRegularData()
		//oa.x.SetGradient(mat.Transpose(mat.MatMulParallel(mat.Transpose(tensor.GradientToMat32()), oa.a.ToMat32f())).Data())
	}
}

func Linear(a, x, b *Tensor) *Tensor {
	//result := Variable(a.shape.X, b.shape.Y)
	//result.op = &opLinear{a, x, b}
	return Add(Matmul(a, x), Expand(b, 0, a.shape.X))
	//return result
}
