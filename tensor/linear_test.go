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
		{"2x2 GPU", Ones(2, 2), Ones(2, 2), Ones(1, 2), true},
		{"2x2 CPU", Ones(2, 2), Ones(2, 2), Ones(1, 2), false},
		{"4x6 and 6x4 + 1x4 GPU", Ones(4, 6), Ones(6, 4), Ones(1, 4), true},
		{"4x6 and 6x4 + 1x4 CPU", Ones(4, 6), Ones(6, 4), Ones(1, 4), false},
		{"1000x784 and 784x128 + 1x128 GPU", Ones(1000, 784), Ones(784, 128), Ones(1, 128), true},
		{"1000x784 and 784x128 + 1x128 CPU", Ones(1000, 784), Ones(784, 128), Ones(1, 128), false},
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

			assert.Equal(t, expected, c.ToFloat32())
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
			c.SetGradient(mat.Ones32f(test.a.Shape().X * test.b.Shape().Y))
			expectedAGrad := mat.MatMulParallel(c.GradientToMat32(), mat.Transpose(test.x.ToMat32f())).Data()
			expectedXGrad := mat.Transpose(mat.MatMulParallel(mat.Transpose(c.GradientToMat32()), test.a.ToMat32f())).Data()
			expectedBGrad := mat.Sum(c.GradientToMat32(), 0).Data()

			c.backward()

			assert.Equal(t, expectedAGrad, test.a.GradientToFloat32())
			assert.Equal(t, expectedXGrad, test.x.GradientToFloat32())
			assert.Equal(t, expectedBGrad, test.b.GradientToFloat32())
		})
	}
}
