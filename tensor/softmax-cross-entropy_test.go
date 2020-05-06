package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

type softmaxCrossEntropyTestCases struct {
	name string
	a *Tensor
	b *Tensor
	runOnGpu bool
}

func buildSoftmaxCrossEntropyTestCases() []softmaxCrossEntropyTestCases {
	return []softmaxCrossEntropyTestCases{
		{"1x1 GPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), true},
		{"1x1 CPU", OfShape(1, 1).SetData([]float32{1}), OfShape(1, 1).SetData([]float32{2}), false},
		{"2x2 GPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), true},
		{"2x2 CPU", OfShape(2, 2).SetData([]float32{1, 2, 3, 4}), OfShape(2, 2).SetData([]float32{5, 6, 7, 8}), false},
	}
}

func Test_softmaxCrossEntropy_forward(t *testing.T) {
	for _, test := range buildSoftmaxCrossEntropyTestCases() {
		t.Run(test.name, func(t *testing.T) {
			t.Log(test.name)

			expected := mat.DivScalar(mat.Sum(mat.Neg(mat.Log(mat.Sum(mat.Mul(test.b.ToMat32f(), mat.Softmax(test.a.ToMat32f())), 1))), 0), float32(test.a.Shape().X)).Data()
			c := SoftmaxCrossEntropy(test.a, test.b)
			c.RunOnGpu(test.runOnGpu)

			c.forward()

			assert.InDeltaSlice(t, expected, c.ToFloat32(), 0.0001)
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
