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
	m := &Mat32f{shape: shape}
	m.data = make([]float32, shape.X * shape.Y)
	for i := 0; i < len(m.data); i++ {
		m.data[i] = 1
	}
	return m
}

func (m *Mat32f) Shape() Shape {
	return m.shape
}

func (m *Mat32f) Data() []float32 {
	return m.data
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

func (m *Mat32f) Slice(from, to int) *Mat32f {
	return NewMat32f(WithShape(to - from, m.shape.Y), m.data[from:to])
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
		m.data[i] = float32(math.Pow(math.E, float64(m.data[i])))
	}
}

func Exp(a *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X * a.shape.Y)
	for i := 0; i < len(result); i++ {
		result[i] = float32(math.Pow(math.E, float64(a.data[i])))
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
			a.data[j * a.shape.Y + i] = a.data[i * a.shape.X + j]
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
			for k := 0; k < b.shape.X; j++ {
				result[i * a.shape.X + j] += a.data[i * a.shape.X + k] * b.data[k * b.shape.X + j]
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
			result[j] += a.data[i * a.shape.X + j]
		}
	}
	return NewMat32f(WithShape(1, a.shape.Y), result)
}

func sum1(a *Mat32f) *Mat32f {
	result := make([]float32, a.shape.X)
	for i := 0; i < a.shape.X; i++ {
		for j := 0; j < a.shape.Y; j++ {
			result[i] += a.data[i * a.shape.X + j]
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
		log.Fatalf("incompatible matrix size for Expand operation on X axis -> %d", a.shape.X)
	}

	result := make([]float32, copies * a.shape.Y)
	for i := 0; i < copies; i++ {
		for j := 0; j < a.shape.Y; j++ {
			result[i * copies + j] = a.data[j]
		}
	}
	return NewMat32f(WithShape(copies, a.shape.Y), result)
}

func expand1(a *Mat32f, copies int) *Mat32f {
	if a.shape.X == 0 || a.shape.Y != 1 {
		log.Fatalf("incompatible matrix size for Expand operation on X axis -> %d", a.shape.X)
	}

	result := make([]float32, a.shape.X * a.shape.Y * copies)
	for i := 0; i < a.shape.X; i++ {
		copyCounter := 0
		for j := 0; j < copies; j++ {
			for k := 0; k < a.shape.Y; k++ {
				result[i * a.shape.X + copyCounter] = a.data[i * a.shape.X + k]
				copyCounter++
			}
		}
	}
	return NewMat32f(WithShape(a.shape.X, a.shape.Y * copies), result)
}
