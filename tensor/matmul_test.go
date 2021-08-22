package tensor

import (
	"github.com/stretchr/testify/assert"
	"ml-framework/mat"
	"testing"
)

type matmulTestCases struct {
	name     string
	a        *Tensor
	b        *Tensor
	runOnGpu bool
}

func buildMatmulTestCases() []matmulTestCases {
	return []matmulTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), false},
		{"64x64 GPU", From(InitRandom, 64, 64), From(InitRandom, 64, 64), true},
		{"64x64 CPU", From(InitRandom, 64, 64), From(InitRandom, 64, 64), false},
		{"50x50 GPU", From(InitRandom, 50, 50), From(InitRandom, 50, 50), true},
		{"50x50 CPU", From(InitRandom, 50, 50), From(InitRandom, 50, 50), false},
		{"4x6 and 6x4 GPU", From(InitRandom, 4, 6), From(InitRandom, 6, 4), true},
		{"4x6 and 6x4 CPU", From(InitRandom, 4, 6), From(InitRandom, 6, 4), false},
		{"4x6 and 6x3 GPU", From(InitRandom, 4, 6), From(InitRandom, 6, 3), true},
		{"4x6 and 6x3 CPU", From(InitRandom, 4, 6), From(InitRandom, 6, 3), false},
		{"1000x784 and 784x128 GPU", From(InitRandom, 1000, 784), From(InitRandom, 784, 128), true},
		{"1000x784 and 784x128 CPU", From(InitRandom, 1000, 784), From(InitRandom, 784, 128), false},
	}
}

func Test_matmul_forward(t *testing.T) {
	for _, test := range buildMatmulTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.MatMul(test.a.ToMat32f(), test.b.ToMat32f()).Data()
			c := Matmul(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.0001)
		})
	}
}

func Test_matmul_backward(t *testing.T) {
	for _, test := range buildMatmulTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := Matmul(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()
			c.SetGradient(mat.Random32f(test.a.Shape().X * test.b.Shape().Y))
			expectedAGrad := mat.MatMul(c.GradientToMat32(), mat.Transpose(test.b.ToMat32f())).Data()
			expectedBGrad := mat.Transpose(mat.MatMul(mat.Transpose(c.GradientToMat32()), test.a.ToMat32f())).Data()

			c.backward()

			assert.InDeltaSlice(t, expectedAGrad, test.a.GradientToFloat32(), 0.0001)
			assert.InDeltaSlice(t, expectedBGrad, test.b.GradientToFloat32(), 0.0001)
		})
	}
}
