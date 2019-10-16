package layer

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/tensor"
	"github.com/stretchr/testify/assert"
	"testing"
)

const (
	someInputSize = 3
)

func Test_Layer_NewLayer_activationFunctionIsSetCorrectly(t *testing.T) {
	l := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	assert.Equal(t, tensor.ActivationFunctionIdentity, l.activationFunctionName)
}

func Test_Layer_NewLayer_inputSizeIsSetCorrectly(t *testing.T) {
	l := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	assert.Equal(t, someInputSize, l.inputSize)
}

func Test_Layer_NewLayer_shouldNotBeAnOutputLayer(t *testing.T) {
	l := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	assert.False(t, l.isOutputLayer)
}

func Test_Layer_NewOutputLayer_shouldBeAnOutputLayer(t *testing.T) {
	l := NewOutputLayer(someInputSize, tensor.ActivationFunctionIdentity)

	assert.True(t, l.isOutputLayer)
}

func Test_Layer_ConnectTo_nextLayerIsSetCorrectly(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	assert.NotNil(t, l1.nextLayer)
	assert.Equal(t,l2, l1.nextLayer)
}

func Test_Layer_ConnectTo_weightsAreInitialized(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	assert.NotNil(t, l1.weights)
}

func Test_Layer_ConnectTo_biasesAreInitialized(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	assert.NotNil(t, l1.bias)
}

func Test_Layer_initializeWeightsAndBias_setsCorrectLengthToWeights(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	assert.Equal(t, l1.inputSize, len(l1.weights.Data()))
}

func Test_Layer_initializeWeightsAndBias_setsCorrectDepthToWeights(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	for _, x := range l1.weights.Data() {
		assert.Equal(t, l2.inputSize, len(x))
	}
}

func Test_Layer_initializeWeightsAndBias_setsCorrectValuesToWeights(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	for _, x := range l1.weights.Data() {
		for _, y := range x {
			assert.NotEqual(t, 0, y)
		}
	}
}

func Test_Layer_initializeWeightsAndBias_setsCorrectLengthToBiases(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	assert.Equal(t, 1, len(l1.bias.Data()))
}

func Test_Layer_initializeWeightsAndBias_setsCorrectDepthToBiases(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	for _, x := range l1.bias.Data() {
		assert.Equal(t, l2.inputSize, len(x))
	}
}

func Test_Layer_initializeWeightsAndBias_setsCorrectValuesToBiases(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	for _, x := range l1.bias.Data() {
		for _, y := range x {
			assert.Equal(t, initialBias, y)
		}
	}
}

func Test_Layer_GetParameters_shouldBeASliceOfTwoElements(t *testing.T) {
	l := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	p := l.GetParameters()

	assert.Equal(t, 2, len(p))
}

func Test_Layer_activate_identityDoesNotChangeTheInput(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	input := generateInput(someInputSize, someInputSize)

	result := l1.activate(input)

	assert.Equal(t, someInputSize, len(result.Data()))
	assert.Equal(t, someInputSize, len(result.Data()[0]))
	assert.Equal(t, input.Data(), result.Data())
}

func Test_Layer_activate_sigmoidCorrectlyModifiesTheInput(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionSigmoid)
	input := generateInput(someInputSize, someInputSize)

	result := l1.activate(input)

	expected := input.Sigmoid()
	assert.Equal(t, expected.Data(), result.Data())
}

func Test_Layer_activate_reluCorrectlyModifiesTheInput(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionRelu)
	input := generateInput(someInputSize, someInputSize)

	result := l1.activate(input)

	expected := input.Relu()
	assert.Equal(t, expected.Data(), result.Data())
}

func Test_Layer_Forward_singleLayerWithIdentityShouldProduceADotOfInputAndWeightsPlusBias(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewOutputLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l1.ConnectTo(l2)
	input := generateInput(someInputSize, someInputSize)

	pred := l1.Forward(input)

	expected := tensor.Add(tensor.Dot(input, l1.weights), tensor.Expand(l1.bias, 0, someInputSize))
	assert.Equal(t, expected.Data(), pred.Data())
}

func Test_Layer_Forward_singleLayerWithSigmoidShouldProduceADotOfActivatedInputAndWeightsPlusBias(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionSigmoid)
	l2 := NewOutputLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l1.ConnectTo(l2)
	input := generateInput(someInputSize, someInputSize)

	pred := l1.Forward(input)

	expected := tensor.Add(tensor.Dot(input.Sigmoid(), l1.weights), tensor.Expand(l1.bias, 0, someInputSize))
	assert.Equal(t, expected.Data(), pred.Data())
}

func Test_Layer_Forward_twoLayersWithIdentityShouldChainTheirResults(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l3 := NewOutputLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l1.ConnectTo(l2)
	l2.ConnectTo(l3)
	input := generateInput(someInputSize, someInputSize)

	pred := l1.Forward(input)

	expected := tensor.Add(tensor.Dot(input, l1.weights), tensor.Expand(l1.bias, 0, someInputSize))
	expected = tensor.Add(tensor.Dot(expected, l2.weights), tensor.Expand(l2.bias, 0, someInputSize))
	assert.Equal(t, expected.Data(), pred.Data())
}

func generateInput(m, n int) *tensor.Tensor {
	mat := mat.CreateMatrix(m, n)
	for i := range mat {
		for j := range mat[i] {
			mat[i][j] = 1
		}
	}
	return tensor.NewTensor(mat)
}