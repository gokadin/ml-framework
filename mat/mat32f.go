package mat

import "math"

type Mat32f struct {
	shape Shape
	data []float32
}

func NewMat32f(shape Shape, data []float32) *Mat32f {
	return &Mat32f{shape, data}
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
