package tensor

import (
	"encoding/json"
	"ml-framework/mat"
	"os"
	"strconv"
)

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR} -lrelu -lsoftmax -llinear -lsoftmaxcrossentropy -lm
//#include <tensor.h>
import "C"

var nextId int

type Tensor struct {
	id                int
	name              string
	op                op
	isGradientEnabled bool
	ready             bool
	_mat_shape        *C.SHAPE
	_grad_shape       *C.SHAPE
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
	t.initializeNativeTensor([]int{1, 1})
	//runtime.SetFinalizer(t, free)

	runOnGpuEnv := os.Getenv("RUN_ON_GPU")
	runOnGpu, err := strconv.ParseBool(runOnGpuEnv)
	if err == nil {
		if !runOnGpu {
			t._tensor.run_on_gpu = C.bool(false)
		}
	}
}

func OfShape(shape ...int) *Tensor {
	if len(shape) != 2 {
		panic("only 2D shapes are supported")
	}

	t := New()
	t.reshapeMat(shape[0], shape[1])
	t.reshapeGrad(shape[0], shape[1])
	return t
}

func Ones(shape ...int) *Tensor {
	t := OfShape(shape...)
	t.SetData(mat.Ones32f(shape[0] * shape[1]))
	return t
}

func Zeros(shape ...int) *Tensor {
	t := OfShape(shape...)
	t.SetData(mat.Zeros32f(shape[0] * shape[1]))
	return t
}

func From(initializer string, shape ...int) *Tensor {
	return initializeParameter(initializer, shape...)
}

func (t *Tensor) initializeNativeTensor(shape []int) {
	t._mat_shape = &C.SHAPE{}
	t._grad_shape = &C.SHAPE{}
	t._tensor.mat_shape = t._mat_shape
	t._tensor.grad_shape = t._grad_shape
	t.reshapeMat(shape[0], shape[1])
	t.reshapeGrad(shape[0], shape[1])
}

func (t *Tensor) reshapeMat(shape ...int) {
	if shape[0] == int(t._mat_shape.x) && shape[1] == int(t._mat_shape.y) {
		return
	}

	t._mat_shape.x = C.int(shape[0])
	t._mat_shape.y = C.int(shape[1])
	t._mat_shape.size = C.int(shape[0] * shape[1])

	t._data = make([]C.float, shape[0]*shape[1])
	t._tensor.data = &t._data[0]
}

func (t *Tensor) reshapeGrad(shape ...int) {
	if shape[0] == int(t._grad_shape.x) && shape[1] == int(t._grad_shape.y) {
		return
	}

	t._grad_shape.x = C.int(shape[0])
	t._grad_shape.y = C.int(shape[1])
	t._grad_shape.size = C.int(shape[0] * shape[1])

	t._grad = make([]C.float, shape[0]*shape[1])
	t._tensor.grad = &t._grad[0]
}

func (t *Tensor) Reshape(shape ...int) *Tensor {
	if len(shape) != 2 {
		panic("only 2D shapes are supported")
	}

	if shape[0] == int(t._mat_shape.x) && shape[1] == int(t._mat_shape.y) {
		return t
	}

	t.reshapeMat(shape...)

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

func (t *Tensor) SetData(data []float32) *Tensor {
	if len(data) > int(t._mat_shape.size) {
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
	result := make([]float32, t.Shape().Size())
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._data[i])
	}
	return result
}

func (t *Tensor) ToFloat64() []float64 {
	result := make([]float64, t._mat_shape.size)
	for i := 0; i < len(result); i++ {
		result[i] = float64(t._data[i])
	}
	return result
}

func (t *Tensor) ToMat32f() *mat.Mat32f {
	result := make([]float32, t._mat_shape.size)
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._data[i])
	}
	return mat.NewMat32f(mat.WithShape(t.Shape().X, t.Shape().Y), result)
}

func (t *Tensor) GradientToFloat32() []float32 {
	result := make([]float32, t._mat_shape.size)
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._grad[i])
	}
	return result
}

func (t *Tensor) GradientToMat32() *mat.Mat32f {
	result := make([]float32, t.Shape().Size())
	for i := 0; i < len(result); i++ {
		result[i] = float32(t._grad[i])
	}
	return mat.NewMat32f(mat.WithShape(int(t._grad_shape.x), int(t._grad_shape.y)), result)
}

func (t *Tensor) Reduce(grad []float32) {
	for i := 0; i < t.Shape().Size(); i++ {
		t._data[i] -= C.float(grad[i])
	}
}

func (t *Tensor) Shape() Shape {
	return Shape{
		X: int(t._mat_shape.x),
		Y: int(t._mat_shape.y),
	}
}

func (t *Tensor) Copy() *Tensor {
	tensor := New()
	tensor.reshapeMat(int(t._mat_shape.x), int(t._mat_shape.y))
	tensor.reshapeGrad(int(t._grad_shape.x), int(t._grad_shape.y))
	tensor.SetData(t.ToFloat32())
	tensor.SetGradient(t.GradientToFloat32())
	return tensor
}

func (t *Tensor) forward() {
	t.reshapeMat(t.op.forwardShape().ToArray()...)
	t.op.forward(t)
	t.ready = true
}

func (t *Tensor) backward() {
	shapes := t.op.backwardShapes(t.Shape())
	for i := 0; i < len(shapes); i++ {
		t.op.dependencies()[i].reshapeGrad(shapes[i].ToArray()...)
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
	t.Reshape(marshaledTensor.Shape.X, marshaledTensor.Shape.Y)
	t.SetData(marshaledTensor.Data)

	return nil
}

type MarshaledTensor struct {
	Name  string    `json:"name"`
	Shape Shape     `json:"shape"`
	Data  []float32 `json:"data"`
}
