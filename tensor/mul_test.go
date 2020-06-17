package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

type mulTestCase struct {
	name string
	a *Tensor
	b *Tensor
	runOnGpu bool
}

func buildMulTestCases() []mulTestCase {
	return []mulTestCase{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), false},
		{"64x64 GPU", From(InitRandom, 64, 64), From(InitRandom, 64, 64), true},
		{"64x64 CPU", From(InitRandom, 64, 64), From(InitRandom, 64, 64), false},
		{"50x50 GPU", From(InitRandom, 50, 50), From(InitRandom, 50, 50), true},
		{"50x50 CPU", From(InitRandom, 50, 50), From(InitRandom, 50, 50), false},
		{"1100x12 GPU", From(InitRandom, 1100, 12), From(InitRandom, 1100, 12), true},
		{"1100x12 CPU", From(InitRandom, 1100, 12), From(InitRandom, 1100, 12), false},
	}
}

func Test_mul_forward(t *testing.T) {
	for _, test := range buildMulTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.Mul(test.a.ToMat32f(), test.b.ToMat32f()).Data()
			c := Mul(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.0001)
		})
	}
}

func Test_mul_backward(t *testing.T) {
	for _, test := range buildMulTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := Mul(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()
			c.SetGradient(mat.Random32f(test.a.Shape().Size()))
			expectedAGrad := mat.Mul(c.GradientToMat32(), test.b.ToMat32f()).Data()
			expectedBGrad := mat.Mul(c.GradientToMat32(), test.a.ToMat32f()).Data()

			c.backward()

			assert.InDeltaSlice(t, expectedAGrad, test.a.GradientToFloat32(), 0.0001)
			assert.InDeltaSlice(t, expectedBGrad, test.b.GradientToFloat32(), 0.0001)
		})
	}
}

