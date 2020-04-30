package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

type addTestCases struct {
	name string
	a *Tensor
	b *Tensor
	runOnGpu bool
}

func buildAddTestCases() []addTestCases {
	return []addTestCases{
		{"1x1 GPU", Variable(1, 1).SetData([]float32{1}), Variable(1, 1).SetData([]float32{2}), true},
		{"1x1 CPU", Variable(1, 1).SetData([]float32{1}), Variable(1, 1).SetData([]float32{2}), false},
		{"2x2 GPU", Variable(2, 2).SetData([]float32{1, 2, 3, 4}), Variable(2, 2).SetData([]float32{5, 6, 7, 8}), true},
		{"2x2 CPU", Variable(2, 2).SetData([]float32{1, 2, 3, 4}), Variable(2, 2).SetData([]float32{5, 6, 7, 8}), false},
	}
}

func Test_add_forward(t *testing.T) {
	t.Parallel()
	for _, test := range buildAddTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			t.Log(test.name)

			expected := mat.Add(test.a.ToMat32f(), test.b.ToMat32f()).Data()
			c := Add(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.Equal(t, expected, c.ToFloat32())
		})
	}
}

func Test_add_backward(t *testing.T) {
	t.Parallel()
	for _, test := range buildAddTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			t.Log(test.name)

			c := Add(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Ones32f(c.Shape().Size()))

			c.backward()

			assert.Equal(t, c.GradientToFloat32(), test.a.GradientToFloat32())
			assert.Equal(t, c.GradientToFloat32(), test.b.GradientToFloat32())
		})
	}
}

func Test_add_forward_invalid_shapes(t *testing.T) {
	a := Variable(1, 2).SetData([]float32{1, 2})
	b := Variable(1, 3).SetData([]float32{1, 2, 3})
	c := Add(a, b)

	assert.Panics(t, c.forward)
}

func Test_add_forward_reshape(t *testing.T) {
	a := Variable(1, 2)
	b := Variable(1, 2)
	c := Add(a, b)
	a.Reshape(2, 2)
	b.Reshape(2, 2)
	a.SetData([]float32{1, 2, 3, 4})
	b.SetData([]float32{1, 2, 3, 4})
	expected := mat.Add(a.ToMat32f(), b.ToMat32f()).Data()

	c.forward()

	assert.Equal(t, a.Shape().ToArray(), c.Shape().ToArray())
	assert.Equal(t, expected, c.ToFloat32())
}

