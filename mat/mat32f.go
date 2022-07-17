package mat

import (
	"log"
	"math"
	"math/rand"
)

type M32f struct {
	shape Shape
	data  []float32
}

func FromSlice32f(shape Shape, data []float32) *M32f {
	return &M32f{shape, data}
}

func Ones32f(size int) []float32 {
	return initialValues32f(size, 1)
}

func Zeros32f(size int) []float32 {
	return initialValues32f(size, 0)
}

func Random32f(size int) []float32 {
	result := make([]float32, size)
	for i := 0; i < size; i++ {
		result[i] = rand.Float32()
	}
	return result
}

func initialValues32f(size int, initialValue float32) []float32 {
	result := make([]float32, size)
	for i := 0; i < size; i++ {
		result[i] = initialValue
	}
	return result
}

func NewMat32fZeros(shape Shape) *M32f {
	return newMat32fWithInitialValue(shape, 0)
}

func newMat32fWithInitialValue(shape Shape, initialValue float32) *M32f {
	m := &M32f{shape: shape}
	m.data = make([]float32, shape.Size())
	for i := 0; i < len(m.data); i++ {
		m.data[i] = initialValue
	}
	return m
}

func (m *M32f) Shape() Shape {
	return m.shape
}

func (m *M32f) Reshape(shape Shape) {
	m.shape = shape
}

func (m *M32f) Data() []float32 {
	return m.data
}

func (m *M32f) Average() float32 {
	var sum float32
	for i := 0; i < len(m.data); i++ {
		sum += m.data[i]
	}
	return sum / float32(len(m.data))
}

func (m *M32f) Copy() *M32f {
	result := make([]float32, len(m.data))
	for i := 0; i < len(m.data); i++ {
		result[i] = m.data[i]
	}
	return FromSlice32f(m.shape, result)
}

func (m *M32f) At(i int) float32 {
	return m.data[i]
}

func (m *M32f) Set(i int, value float32) {
	m.data[i] = value
}

func (m *M32f) Apply(mapping func(float32) float32) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = mapping(m.data[i])
	}
}

func Apply(a *M32f, mapping func(float32) float32) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(a.data); i++ {
		result[i] = mapping(a.data[i])
	}
	return FromSlice32f(a.shape, result)
}

/*
	mat of shape (3, 4)
	[[11 12 13 14]
	 [21 22 23 24]
	 [31 32 33 34]]
	Slice with begin (1, 1), size (2, 3)
	produces:
	[[22 23 24]
	 [32 33 34]]

	mat of shape (2, 3, 4)
	[
		[[11 12 13 14]
		 [21 22 23 24]
		 [31 32 33 34]]

		[[11 12 13 14]
		 [21 22 23 24]
		 [31 32 33 34]]
	]
	Slice with begin (0, 2, 2), size (2, 1, 1)
	produces
	[
		[[33]]

		[[33]]
	]
*/
func (m *M32f) Slice(begin, size Shape) *M32f {
	if begin.Count() != size.Count() {
		panic("shape dimension count must match in order to perform a slice")
	}

	for i := 0; i < size.Count(); i++ {
		size.D[i] -= 1
	}

	sliced := m.data[m.shape.IndexOf(begin.D...):m.shape.IndexOf(AddShapes(begin, size).D...)]
	return FromSlice32f(size, sliced)
}

func (m *M32f) Add(other *M32f) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] += other.data[i]
	}
}

func Add(a, b *M32f) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] + b.data[i]
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) AddScalar(scalar float32) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] += scalar
	}
}

func AddScalar(a *M32f, scalar float32) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] + scalar
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) Sub(other *M32f) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] -= other.data[i]
	}
}

func Sub(a, b *M32f) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] - b.data[i]
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) Div(other *M32f) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] /= other.data[i]
	}
}

func Div(a, b *M32f) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] / b.data[i]
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) DivScalarBy(scalar float32) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = scalar / m.data[i]
	}
}

func DivScalarBy(a *M32f, scalar float32) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = scalar / a.data[i]
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) DivScalar(scalar float32) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = m.data[i] / scalar
	}
}

func DivScalar(a *M32f, scalar float32) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] / scalar
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) SubFromScalar(scalar float32) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = scalar - m.data[i]
	}
}

func SubFromScalar(a *M32f, scalar float32) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = scalar - a.data[i]
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) Mul(other *M32f) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = m.data[i] * other.data[i]
	}
}

func Mul(a, b *M32f) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] * b.data[i]
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) MulScalar(scalar float32) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = m.data[i] * scalar
	}
}

func MulScalar(a *M32f, scalar float32) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] * scalar
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) Sqrt() {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = float32(math.Sqrt(float64(m.data[i])))
	}
}

func Sqrt(a *M32f) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = float32(math.Sqrt(float64(a.data[i])))
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) Pow(power float64) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = float32(math.Pow(float64(m.data[i]), power))
	}
}

func Pow(a *M32f, power float64) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = float32(math.Pow(float64(a.data[i]), power))
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) Exp() {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = float32(math.Exp(float64(m.data[i])))
	}
}

func Exp(a *M32f) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = float32(math.Exp(float64(a.data[i])))
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) Log() {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = float32(math.Log(float64(m.data[i])))
	}
}

func Log(a *M32f) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = float32(math.Log(float64(a.data[i])))
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) Neg() {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = -m.data[i]
	}
}

func Neg(a *M32f) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < len(result); i++ {
		result[i] = -a.data[i]
	}
	return FromSlice32f(a.shape, result)
}

func (m *M32f) Transpose() {
	for i := 0; i < m.shape.D[0]; i++ {
		for j := 0; j < m.shape.D[1]; j++ {
			m.data[j*m.shape.D[1]+i] = m.data[i*m.shape.D[0]+j]
		}
	}
	temp := m.shape.D[0]
	m.shape.D[0] = m.shape.D[1]
	m.shape.D[1] = temp
}

func Transpose(a *M32f) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < a.shape.D[0]; i++ {
		for j := 0; j < a.shape.D[1]; j++ {
			result[j*a.shape.D[0]+i] = a.data[i*a.shape.D[1]+j]
		}
	}
	return FromSlice32f(Dim(a.shape.D[1], a.shape.D[0]), result)
}

func MatMul(a, b *M32f) *M32f {
	if a.shape.D[0] == 0 || a.shape.D[1] == 0 || b.shape.D[0] == 0 || b.shape.D[1] == 0 || a.shape.D[1] != b.shape.D[0] {
		log.Fatalf("cannot multiply matrices of incompatible sizes -> %dx%d and %dx%d", a.shape.D[0], a.shape.D[1], b.shape.D[0], b.shape.D[1])
	}

	result := make([]float32, a.shape.D[0]*b.shape.D[1])
	for i := 0; i < a.shape.D[0]; i++ {
		for j := 0; j < b.shape.D[1]; j++ {
			for k := 0; k < b.shape.D[0]; k++ {
				result[i*b.shape.D[1]+j] += a.data[i*b.shape.D[0]+k] * b.data[k*b.shape.D[1]+j]
			}
		}
	}
	return FromSlice32f(Dim(a.shape.D[0], b.shape.D[1]), result)
}

func Sum(a *M32f, axis int) *M32f {
	switch axis {
	case 0:
		return sum0(a)
	case 1:
		return sum1(a)
	default:
		log.Fatal("sum only supports axis 0 and 1")
		return nil
	}
}

/**
Shape of (A,B) becomes (1,B)
*/
func sum0(a *M32f) *M32f {
	result := make([]float32, a.shape.D[1])
	for i := 0; i < a.shape.D[0]; i++ {
		for j := 0; j < a.shape.D[1]; j++ {
			result[j] += a.data[i*a.shape.D[1]+j]
		}
	}
	return FromSlice32f(Dim(1, a.shape.D[1]), result)
}

/**
Shape of (A,B) becomes (A,1)
*/
func sum1(a *M32f) *M32f {
	result := make([]float32, a.shape.D[0])
	for i := 0; i < a.shape.D[0]; i++ {
		for j := 0; j < a.shape.D[1]; j++ {
			result[i] += a.data[i*a.shape.D[1]+j]
		}
	}
	return FromSlice32f(Dim(a.shape.D[0], 1), result)
}

func Expand(a *M32f, axis, copies int) *M32f {
	switch axis {
	case 0:
		return expand0(a, copies)
	case 1:
		return expand1(a, copies)
	default:
		log.Fatal("expand only supports axis 0 and 1")
		return nil
	}
}

/**
Shape of (A,B) becomes (A*copies,B)
*/
func expand0(a *M32f, copies int) *M32f {
	if a.shape.D[0] != 1 || a.shape.Count() > 2 {
		log.Fatalf("incompatible matrix size for Expand operation on X axis -> %dx%d", a.shape.D[0], a.shape.D[1])
	}

	result := make([]float32, copies*a.shape.D[1])
	for i := 0; i < copies; i++ {
		for j := 0; j < a.shape.D[1]; j++ {
			result[i*a.shape.D[1]+j] = a.data[j]
		}
	}
	return FromSlice32f(Dim(copies, a.shape.D[1]), result)
}

/**
Shape of (A,B) becomes (A,B*copies)
*/
func expand1(a *M32f, copies int) *M32f {
	if a.shape.D[0] == 0 || a.shape.D[1] != 1 || a.shape.Count() > 2 {
		log.Fatalf("incompatible matrix size for Expand operation on Y axis -> %dx%d", a.shape.D[0], a.shape.D[1])
	}

	result := make([]float32, a.shape.D[0]*a.shape.D[1]*copies)
	for i := 0; i < a.shape.D[0]; i++ {
		for j := 0; j < copies; j++ {
			result[i*copies+j] = a.data[i]
		}
	}
	return FromSlice32f(Dim(a.shape.D[0], copies), result)
}

func (m *M32f) Equals32f(other *M32f) bool {
	return Equals32f(m, other)
}

func Equals32f(a, b *M32f) bool {
	if a == nil || b == nil || !a.shape.Equals(b.shape) {
		return false
	}

	for i := 0; i < len(a.data); i++ {
		if math.Abs(float64(a.data[i]-b.data[i])) > 1e-6 {
			return false
		}
	}

	return true
}

func Softmax(a *M32f) *M32f {
	result := make([]float32, a.shape.Size())
	for i := 0; i < a.shape.D[0]; i++ {
		sum := 0.0
		for j := 0; j < a.shape.D[1]; j++ {
			sum += math.Exp(float64(a.data[i*a.shape.D[1]+j]))
		}
		for j := 0; j < a.shape.D[1]; j++ {
			index := i*a.shape.D[1] + j
			result[index] = float32(math.Exp(float64(a.data[index])) / sum)
		}
	}
	return FromSlice32f(a.shape, result)
}
