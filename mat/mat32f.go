package mat

import (
	"log"
	"math"
)

type Mat32f struct {
	shape Shape
	data []float32
}

func NewMat32f(shape Shape, data []float32) *Mat32f {
	return &Mat32f{shape, data}
}

func NewMat32fOnes(shape Shape) *Mat32f {
	return newMat32fWithInitialValue(shape, 1)
}

func NewMat32fZeros(shape Shape) *Mat32f {
	return newMat32fWithInitialValue(shape, 0)
}

func newMat32fWithInitialValue(shape Shape, initialValue float32) *Mat32f {
	m := &Mat32f{shape: shape}
	m.data = make([]float32, shape.X * shape.Y)
	for i := 0; i < len(m.data); i++ {
		m.data[i] = initialValue
	}
	return m
}

func (m *Mat32f) Shape() Shape {
	return m.shape
}

func (m *Mat32f) Reshape(shape Shape) {
	m.shape = shape
}

func (m *Mat32f) Data() []float32 {
	return m.data
}

func (m *Mat32f) ToFloat64() []float64 {
	result := make([]float64, len(m.data))
	for i := 0; i < len(m.data); i++ {
		result[i] = float64(m.data[i])
	}
	return result
}

func (m *Mat32f) Copy() []float32 {
	result := make([]float32, len(m.data))
	for i := 0; i < len(m.data); i++ {
		result[i] = m.data[i]
	}
	return result
}

func (m *Mat32f) At(i int) float32 {
	return m.data[i]
}

func (m *Mat32f) Set(i int, value float32) {
	m.data[i] = value
}

func (m *Mat32f) Apply(mapping func(float32) float32) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = mapping(m.data[i])
	}
}

func Apply(a *Mat32f, mapping func(float32) float32) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(a.data); i++ {
		result[i] = mapping(a.data[i])
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) Slice(from, to int) *Mat32f {
	shapeX := to - from
	from *= m.shape.Y
	to *= m.shape.Y
	return NewMat32f(WithShape(shapeX, m.shape.Y), m.data[from:to])
}

func (m *Mat32f) Add(other *Mat32f) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] += other.data[i]
	}
}

func Add(a, b *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] + b.data[i]
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) AddScalar(scalar float32) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] += scalar
	}
}

func AddScalar(a *Mat32f, scalar float32) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] + scalar
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) Sub(other *Mat32f) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] -= other.data[i]
	}
}

func Sub(a, b *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] - b.data[i]
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) Div(other *Mat32f) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] /= other.data[i]
	}
}

func Div(a, b *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] / b.data[i]
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) DivScalarBy(scalar float32) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = scalar / m.data[i]
	}
}

func DivScalarBy(a *Mat32f, scalar float32) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = scalar / a.data[i]
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) DivScalar(scalar float32) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = m.data[i] / scalar
	}
}

func DivScalar(a *Mat32f, scalar float32) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] / scalar
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) SubFromScalar(scalar float32) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = scalar - m.data[i]
	}
}

func SubFromScalar(a *Mat32f, scalar float32) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = scalar - a.data[i]
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) Mul(other *Mat32f) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = m.data[i] * other.data[i]
	}
}

func Mul(a, b *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] * b.data[i]
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) MulScalar(scalar float32) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = m.data[i] * scalar
	}
}

func MulScalar(a *Mat32f, scalar float32) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = a.data[i] * scalar
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) Sqrt() {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = float32(math.Sqrt(float64(m.data[i])))
	}
}

func Sqrt(a *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = float32(math.Sqrt(float64(a.data[i])))
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) Pow(power float64) {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = float32(math.Pow(float64(m.data[i]), power))
	}
}

func Pow(a *Mat32f, power float64) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = float32(math.Pow(float64(a.data[i]), power))
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) Exp() {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = float32(math.Exp(float64(m.data[i])))
	}
}

func Exp(a *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = float32(math.Exp(float64(a.data[i])))
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) Log() {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = float32(math.Log(float64(m.data[i])))
	}
}

func Log(a *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = float32(math.Log(float64(a.data[i])))
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) Neg() {
	for i := 0; i < len(m.data); i++ {
		m.data[i] = -m.data[i]
	}
}

func Neg(a *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = -a.data[i]
	}
	return NewMat32f(a.shape, result)
}

func (m *Mat32f) Transpose() {
	for i := 0; i < m.shape.X; i++ {
		for j := 0; j < m.shape.Y; j++ {
			m.data[j * m.shape.Y + i] = m.data[i * m.shape.X + j]
		}
	}
	temp := m.shape.X
	m.shape.X = m.shape.Y
	m.shape.Y = temp
}

func Transpose(a *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < a.shape.X; i++ {
		for j := 0; j < a.shape.Y; j++ {
			result[j * a.shape.X + i] = a.data[i * a.shape.Y + j]
		}
	}
	return NewMat32f(WithShape(a.shape.Y, a.shape.X), result)
}

func MatMul(a, b *Mat32f) *Mat32f {
	if a.shape.X == 0 || a.shape.Y == 0 || b.shape.X == 0 || b.shape.Y == 0 || a.shape.Y != b.shape.X {
		log.Fatalf("cannot multiply matrices of incompatible sizes -> %dx%d and %dx%d", a.shape.X, a.shape.Y, b.shape.X, b.shape.Y)
	}

	result := make([]float32, a.shape.X * b.shape.Y)
	for i := 0; i < a.shape.X; i++ {
		for j := 0; j < b.shape.Y; j++ {
			for k := 0; k < b.shape.X; k++ {
				result[i * b.shape.Y + j] += a.data[i * b.shape.X + k] * b.data[k * b.shape.Y + j]
			}
		}
	}
	return NewMat32f(WithShape(a.shape.X, b.shape.Y), result)
}

func Sum(a *Mat32f, axis int) *Mat32f {
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

func sum0(a *Mat32f) *Mat32f {
	result := make([]float32, a.shape.Y)
	for i := 0; i < a.shape.X; i++ {
		for j := 0; j < a.shape.Y; j++ {
			result[j] += a.data[i * a.shape.Y + j]
		}
	}
	return NewMat32f(WithShape(1, a.shape.Y), result)
}

func sum1(a *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X)
	for i := 0; i < a.shape.X; i++ {
		for j := 0; j < a.shape.Y; j++ {
			result[i] += a.data[i * a.shape.Y + j]
		}
	}
	return NewMat32f(WithShape(a.shape.X, 1), result)
}

func Expand(a *Mat32f, axis, copies int) *Mat32f {
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

func expand0(a *Mat32f, copies int) *Mat32f {
	if a.shape.X != 1 {
		log.Fatalf("incompatible matrix size for Expand operation on X axis -> %dx%d", a.shape.X, a.shape.Y)
	}

	result := make([]float32, copies * a.shape.Y)
	for i := 0; i < copies; i++ {
		for j := 0; j < a.shape.Y; j++ {
			result[i * a.shape.Y + j] = a.data[j]
		}
	}
	return NewMat32f(WithShape(copies, a.shape.Y), result)
}

func expand1(a *Mat32f, copies int) *Mat32f {
	if a.shape.X == 0 || a.shape.Y != 1 {
		log.Fatalf("incompatible matrix size for Expand operation on Y axis -> %dx%d", a.shape.X, a.shape.Y)
	}

	result := make([]float32, a.shape.X * a.shape.Y * copies)
	for i := 0; i < a.shape.X; i++ {
		for j := 0; j < copies; j++ {
			result[i * copies + j] = a.data[i * a.shape.Y]
		}
	}
	return NewMat32f(WithShape(a.shape.X, a.shape.Y * copies), result)
}

func (m *Mat32f) Equals32f(other *Mat32f) bool {
	return Equals32f(m, other)
}

func Equals32f(a, b *Mat32f) bool {
	if a == nil || b == nil || a.shape.X != b.shape.X || a.shape.Y != b.shape.Y {
		return false
	}

	for i := 0; i < len(a.data); i++ {
		if a.data[i] != b.data[i] {
			return false
		}
	}

	return true
}

func Softmax(a *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < a.shape.X; i++ {
		sum := 0.0
		for j := 0; j < a.shape.Y; j++ {
			sum += math.Exp(float64(a.data[i * a.shape.Y + j]))
		}
		for j := 0; j < a.shape.Y; j++ {
			index := i * a.shape.Y + j
			result[index] = float32(math.Exp(float64(a.data[index])) / sum)
		}
	}
	return NewMat32f(a.shape, result)
}
