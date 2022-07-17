package tensor

import (
	"encoding/json"
	"ml-framework/mat"
	"os"
	"strconv"
)

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR}/../lib -Wl,-rpath,${SRCDIR}/../lib -lrelu -lsoftmax -llinear -lsoftmaxcrossentropy -lm
//#include <tensor.h>
import "C"

var nextId int

type Tensor struct {
	id                int
	name              string
	op                op
	isGradientEnabled bool
	ready             bool
	_mat_shape        []C.int
	_grad_shape       []C.int
	_data             []C.float
	_grad             []C.float
	_tensor           *C.TENSOR
}

func New() *Tensor {
	nextId++
	t := &Tensor{
		id: nextId,
	}

	t.init()

	return t
}

func (t *Tensor) init() {
	t._tensor = C.alloc_tensor(C.int(nextId))
	t.initializeNativeTensor(mat.Dim(1))
	//runtime.SetFinalizer(t, free)

	runOnGpuEnv := os.Getenv("RUN_ON_GPU")
	runOnGpu, err := strconv.ParseBool(runOnGpuEnv)
	if err == nil {
		if !runOnGpu {
			t._tensor.run_on_gpu = C.bool(false)
		}
	}
}

func OfShape(dimensions ...int) *Tensor {
	shape := mat.Dim(dimensions...)
	t := New()
	t.reshapeMat(shape)
	t.reshapeGrad(shape)
	return t
}

func FromMat32(m *mat.M32f) *Tensor {
	t := OfShape(m.Shape().D...)
	t.SetData(m.Data())
	return t
}

func Ones(dimensions ...int) *Tensor {
	shape := mat.Dim(dimensions...)
	t := OfShape(dimensions...)
	t.SetData(mat.Ones32f(shape.Size()))
	return t
}

func Zeros(dimensions ...int) *Tensor {
	shape := mat.Dim(dimensions...)
	t := OfShape(dimensions...)
	t.SetData(mat.Zeros32f(shape.Size()))
	return t
}

func From(initializer string, shape ...int) *Tensor {
	return FromMat32(mat.Initialize(initializer, mat.Dim(shape...)))
}

func (t *Tensor) initializeNativeTensor(shape mat.Shape) {
	t.reshapeMat(shape)
	t.reshapeGrad(shape)
}

func (t *Tensor) reshapeMat(shape mat.Shape) {
	t._mat_shape = toNativeArrayInt(shape.D)
	t._tensor.mat_shape = &t._mat_shape[0]
	t._tensor.mat_size = C.int(shape.Size())

	t._data = make([]C.float, shape.Size())
	t._tensor.data = &t._data[0]
}

func (t *Tensor) reshapeGrad(shape mat.Shape) {
	t._grad_shape = toNativeArrayInt(shape.D)
	t._tensor.grad_shape = &t._grad_shape[0]
	t._tensor.grad_size = C.int(shape.Size())

	t._grad = make([]C.float, shape.Size())
	t._tensor.grad = &t._grad[0]
}

func (t *Tensor) Reshape(dimensions ...int) *Tensor {
	shape := mat.Dim(dimensions...)
	t.reshapeMat(shape)

	return t
}

func free(t *Tensor) {
	C.free_tensor(t._tensor)
}

func (t *Tensor) Id() int {
	return t.id
}

func (t *Tensor) Name() string {
	return t.name
}

func (t *Tensor) SetName(name string) *Tensor {
	t.name = name
	return t
}

func (t *Tensor) RunOnGpu(value bool) {
	t._tensor.run_on_gpu = C.bool(value)
}

func (t *Tensor) Size() int {
	result := 1
	for _, d := range t._mat_shape {
		result *= int(d)
	}
	return result
}

func (t *Tensor) SetData(data []float32) *Tensor {
	if len(data) > t.Size() {
		panic("given data is bigger than tensor size")
	}

	for i := 0; i < len(data); i++ {
		t._data[i] = C.float(data[i])
	}
	return t
}

func (t *Tensor) Set(index int, value float32) {
	t._data[index] = C.float(value)
}

func (t *Tensor) SetGradient(grad []float32) {
	for i := 0; i < len(grad); i++ {
		t._grad[i] = C.float(grad[i])
	}
}

func (t *Tensor) ToFloat32() []float32 {
	result := make([]float32, t.Size())
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._data[i])
	}
	return result
}

func (t *Tensor) ToFloat64() []float64 {
	result := make([]float64, t.Size())
	for i := 0; i < len(result); i++ {
		result[i] = float64(t._data[i])
	}
	return result
}

func (t *Tensor) ToMat32f() *mat.M32f {
	result := make([]float32, t.Size())
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._data[i])
	}
	return mat.FromSlice32f(t.Shape(), result)
}

func (t *Tensor) GradientToFloat32() []float32 {
	result := make([]float32, t.Size())
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._grad[i])
	}
	return result
}

func (t *Tensor) GradientToMat32() *mat.M32f {
	result := fromNativeArrayF32(t._grad)
	return mat.FromSlice32f(t.GradShape(), result)
}

func (t *Tensor) Reduce(grad []float32) {
	for i := 0; i < t.Size(); i++ {
		t._data[i] -= C.float(grad[i])
	}
}

func (t *Tensor) Shape() mat.Shape {
	return mat.Dim(fromNativeArrayInt(t._mat_shape)...)
}

func (t *Tensor) GradShape() mat.Shape {
	return mat.Dim(fromNativeArrayInt(t._grad_shape)...)
}

func (t *Tensor) Copy() *Tensor {
	tensor := New()
	tensor.reshapeMat(tensor.Shape())
	tensor.reshapeGrad(tensor.GradShape())
	tensor.SetData(t.ToFloat32())
	tensor.SetGradient(t.GradientToFloat32())
	return tensor
}

func (t *Tensor) forward() {
	t.reshapeMat(t.op.forwardShape())
	t.op.forward(t)
	t.ready = true
}

func (t *Tensor) backward() {
	shapes := t.op.backwardShapes(t.Shape())
	for i := 0; i < len(shapes); i++ {
		t.op.dependencies()[i].reshapeGrad(shapes[i])
	}
	t.op.backward(t)
	t.ready = false
}

func (t *Tensor) MarshalJSON() ([]byte, error) {
	return json.Marshal(MarshaledTensor{
		Name:  t.name,
		Shape: t.Shape(),
		Data:  t.ToFloat32(),
	})
}

func (t *Tensor) UnmarshalJSON(b []byte) error {
	marshaledTensor := MarshaledTensor{}
	json.Unmarshal(b, &marshaledTensor)

	t.init()
	t.Reshape(marshaledTensor.Shape.D...)
	t.SetData(marshaledTensor.Data)

	return nil
}

type MarshaledTensor struct {
	Name  string    `json:"name"`
	Shape mat.Shape `json:"shape"`
	Data  []float32 `json:"data"`
}
