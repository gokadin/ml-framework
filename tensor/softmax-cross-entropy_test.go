package tensor

import (
	"github.com/stretchr/testify/assert"
	"ml-framework/mat"
	"testing"
)

type softmaxCrossEntropyTestCases struct {
	name     string
	a        *Tensor
	b        *Tensor
	runOnGpu bool
}

func buildSoftmaxCrossEntropyTestCases() []softmaxCrossEntropyTestCases {
	return []softmaxCrossEntropyTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), false},
		{"5x2 GPU", From(InitRandom, 5, 2), From(InitRandom, 5, 2), true},
		{"5x2 CPU", From(InitRandom, 5, 2), From(InitRandom, 5, 2), false},
		{"2x5 GPU", From(InitRandom, 2, 5), From(InitRandom, 2, 5), true},
		{"2x5 CPU", From(InitRandom, 2, 5), From(InitRandom, 2, 5), false},
		{"1000x10 GPU", From(InitRandom, 1000, 10), From(InitRandom, 1000, 10), true},
		{"1000x10 CPU", From(InitRandom, 1000, 10), From(InitRandom, 1000, 10), false},
	}
}

func Test_softmaxCrossEntropy_forward(t *testing.T) {
	for _, test := range buildSoftmaxCrossEntropyTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			aSoftmaxMat := mat.Softmax(test.a.ToMat32f())
			expected := mat.DivScalar(mat.Sum(mat.Neg(mat.Log(mat.Sum(mat.Mul(test.b.ToMat32f(), aSoftmaxMat), 1))), 0), float32(test.a.Shape().X)).Data()
			c := SoftmaxCrossEntropy(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.0001)
			assert.InDeltaSlice(t, aSoftmaxMat.Data(), test.a.ToFloat32(), 0.0001)
		})
	}
}

func Test_softmaxCrossEntropy_backward(t *testing.T) {
	for _, test := range buildSoftmaxCrossEntropyTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			c := SoftmaxCrossEntropy(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)
			c.forward()
			c.SetGradient(mat.Random32f(c.Shape().Size()))
			expandedGrad := mat.Expand(mat.Expand(c.GradientToMat32(), 1, test.a.Shape().Y), 0, test.a.Shape().X)
			expectedGrad := mat.Mul(expandedGrad, mat.Sub(test.a.ToMat32f(), test.b.ToMat32f())).Data()

			c.backward()

			assert.InDeltaSlice(t, expectedGrad, test.a.GradientToFloat32(), 0.0001)
		})
	}
}
