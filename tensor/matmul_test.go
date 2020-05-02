package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

type matmulTestCases struct {
	name string
	a *Tensor
	b *Tensor
	runOnGpu bool
}

func buildMatmulTestCases() []matmulTestCases {
	return []matmulTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), false},
		{"64x64 GPU", Ones(64, 64), Ones(64, 64), true},
		{"64x64 CPU", Ones(64, 64), Ones(64, 64), false},
		{"50x50 GPU", Ones(50, 50), Ones(50, 50), true},
		{"50x50 CPU", Ones(50, 50), Ones(50, 50), false},
		{"4x6 and 6x4 GPU", Ones(4, 6), Ones(6, 4), true},
		{"4x6 and 6x4 CPU", Ones(4, 6), Ones(6, 4), false},
		{"4x6 and 6x3 GPU", Ones(4, 6), Ones(6, 3), true},
		{"4x6 and 6x3 CPU", Ones(4, 6), Ones(6, 3), false},
		{"1000x784 and 784x128 GPU", Ones(1000, 784), Ones(784, 128), true},
		{"1000x784 and 784x128 CPU", Ones(1000, 784), Ones(784, 128), false},
	}
}

func Test_matmul_forward(t *testing.T) {
	t.Parallel()
	for _, test := range buildMatmulTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			t.Log(test.name)

			expected := mat.MatMul(test.a.ToMat32f(), test.b.ToMat32f()).Data()
			c := Matmul(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.Equal(t, expected, c.ToFloat32())
		})
	}
}

func Test_matmul_backward(t *testing.T) {
	t.Parallel()
	for _, test := range buildMatmulTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			t.Log(test.name)

			c := Matmul(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()
			c.SetGradient(mat.Ones32f(test.a.Shape().X * test.b.Shape().Y))
			expectedAGrad := mat.MatMul(c.GradientToMat32(), mat.Transpose(test.b.ToMat32f())).Data()
			expectedBGrad := mat.Transpose(mat.MatMul(mat.Transpose(c.GradientToMat32()), test.a.ToMat32f())).Data()

			c.backward()

			assert.Equal(t, expectedAGrad, test.a.GradientToFloat32())
			assert.Equal(t, expectedBGrad, test.b.GradientToFloat32())
		})
	}
}
