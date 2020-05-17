package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

type linearTestCases struct {
	name string
	a *Tensor
	x *Tensor
	b *Tensor
	runOnGpu bool
}

func buildLinearTestCases() []linearTestCases {
	return []linearTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), OfShape(1, 1).SetData([]float32{3}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), OfShape(1, 1).SetData([]float32{3}), false},
		{"2x2 GPU", From(InitRandom, 2, 2), From(InitRandom, 2, 2), From(InitRandom, 1, 2), true},
		{"2x2 CPU", From(InitRandom, 2, 2), From(InitRandom, 2, 2), From(InitRandom, 1, 2), false},
		{"4x6 and 6x4 + 1x4 GPU", From(InitRandom, 4, 6), From(InitRandom, 6, 4), From(InitRandom, 1, 4), true},
		{"4x6 and 6x4 + 1x4 CPU", From(InitRandom, 4, 6), From(InitRandom, 6, 4), From(InitRandom, 1, 4), false},
		{"1000x784 and 784x128 + 1x128 GPU", From(InitRandom, 1000, 784), From(InitRandom, 784, 128), From(InitRandom, 1, 128), true},
		{"1000x784 and 784x128 + 1x128 CPU", From(InitRandom, 1000, 784), From(InitRandom, 784, 128), From(InitRandom, 1, 128), false},
	}
}

func Test_linear_forward(t *testing.T) {
	for _, test := range buildLinearTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.Add(mat.MatMul(test.a.ToMat32f(), test.x.ToMat32f()), mat.Expand(test.b.ToMat32f(), 0, test.a.Shape().X)).Data()
			c := Linear(test.a, test.x, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.0001)
		})
	}
}

func Test_linear_backward(t *testing.T) {
	for _, test := range buildLinearTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := Linear(test.a, test.x, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()
			c.SetGradient(mat.Random32f(test.a.Shape().X * test.b.Shape().Y))
			expectedAGrad := mat.MatMulParallel(c.GradientToMat32(), mat.Transpose(test.x.ToMat32f())).Data()
			expectedXGrad := mat.Transpose(mat.MatMulParallel(mat.Transpose(c.GradientToMat32()), test.a.ToMat32f())).Data()
			expectedBGrad := mat.Sum(c.GradientToMat32(), 0).Data()

			c.backward()

			assert.InDeltaSlice(t, expectedAGrad, test.a.GradientToFloat32(), 0.0001)
			assert.InDeltaSlice(t, expectedXGrad, test.x.GradientToFloat32(), 0.0001)
			assert.InDeltaSlice(t, expectedBGrad, test.b.GradientToFloat32(), 0.0001)
		})
	}
}

func Test_linear_forward_invalid_shapes_between_a_x(t *testing.T) {
	a := OfShape(1, 2)
	x := OfShape(1, 2)
	b := OfShape(1, 2)
	c := Linear(a, x, b)

	assert.Panics(t, c.forward)
}

func Test_linear_forward_invalid_shapes_for_b(t *testing.T) {
	a := OfShape(1, 2)
	x := OfShape(2, 3)
	b := OfShape(1, 2)
	c := Linear(a, x, b)

	assert.Panics(t, c.forward)
}

func Test_linear_forward_reshape(t *testing.T) {
	a := From(InitRandom, 1, 2)
	x := From(InitRandom, 2, 3)
	b := From(InitRandom, 1, 3)
	c := Linear(a, x, b)
	expected := mat.Add(mat.MatMul(a.ToMat32f(), x.ToMat32f()), mat.Expand(b.ToMat32f(), 0, a.Shape().X)).Data()

	c.forward()

	assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.000001)

	a.Reshape(3, 2).SetData([]float32{1, 2, 3, 4, 5, 6})
	expected = mat.Add(mat.MatMul(a.ToMat32f(), x.ToMat32f()), mat.Expand(b.ToMat32f(), 0, a.Shape().X)).Data()

	c.forward()

	assert.Equal(t, []int{3, 3}, c.Shape().ToArray())
	assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.000001)
}

