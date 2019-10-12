package layer

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/tensor"
	"testing"
)

const (
	someInputSize = 3
)

func Test_Layer_NewLayer_activationFunctionIsSetCorrectly(t *testing.T) {
	l := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	if l.activationFunctionName != tensor.ActivationFunctionIdentity {
		t.Fatal("activation function was not correctly initialized")
	}
}

func Test_Layer_NewLayer_inputSizeIsSetCorrectly(t *testing.T) {
	l := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	if l.inputSize != someInputSize {
		t.Fatal("input size was not correctly initialized")
	}
}

func Test_Layer_NewLayer_shouldNotBeAnOutputLayer(t *testing.T) {
	l := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	if l.isOutputLayer {
		t.Fatal("layer should not be an output layer")
	}
}

func Test_Layer_NewOutputLayer_shouldBeAnOutputLayer(t *testing.T) {
	l := NewOutputLayer(someInputSize, tensor.ActivationFunctionIdentity)

	if !l.isOutputLayer {
		t.Fatal("layer should be an output layer")
	}
}

func Test_Layer_ConnectTo_nextLayerIsSetCorrectly(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	if l1.nextLayer == nil || l1.nextLayer != l2 {
		t.Fatal("next layer is not set correctly")
	}
}

func Test_Layer_ConnectTo_weightsAreInitialized(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	if l1.weights == nil {
		t.Fatal("weights should be initialized")
	}
}

func Test_Layer_ConnectTo_biasesAreInitialized(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	if l1.bias == nil {
		t.Fatal("weights should be initialized")
	}
}

func Test_Layer_initializeWeightsAndBias_setsCorrectLengthToWeights(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	if len(l1.weights.Data()) != l1.inputSize {
		t.Fatal("weights length is incorrect")
	}
}

func Test_Layer_initializeWeightsAndBias_setsCorrectDepthToWeights(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	for _, x := range l1.weights.Data() {
		if len(x) != l2.inputSize {
			t.Fatal("weights depth is incorrect")
		}
	}
}

func Test_Layer_initializeWeightsAndBias_setsCorrectValuesToWeights(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	for _, x := range l1.weights.Data() {
		for _, y := range x {
			if y == 0 {
				t.Fatal("weight value should not be zero")
			}
		}
	}
}

func Test_Layer_initializeWeightsAndBias_setsCorrectLengthToBiases(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	if len(l1.bias.Data()) != 1 {
		t.Fatal("biases length is incorrect")
	}
}

func Test_Layer_initializeWeightsAndBias_setsCorrectDepthToBiases(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	for _, x := range l1.bias.Data() {
		if len(x) != l2.inputSize {
			t.Fatal("biases depth is incorrect")
		}
	}
}

func Test_Layer_initializeWeightsAndBias_setsCorrectValuesToBiases(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	l1.ConnectTo(l2)

	for _, x := range l1.bias.Data() {
		for _, y := range x {
			if y != initialBias {
				t.Fatal("bias value is incorrect")
			}
		}
	}
}

func Test_Layer_GetParameters_shouldBeASliceOfTwoElements(t *testing.T) {
	l := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)

	p := l.GetParameters()

	if len(p) != 2 {
		t.Fatal("slice should contain two elements")
	}
}

func Test_Layer_activate_identityDoesNotChangeTheInput(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	input := generateInput(someInputSize, someInputSize)

	result := l1.activate(input)

	if len(result.Data()) != someInputSize || len(result.Data()[0]) != someInputSize {
		t.Fatal("result dimensions changed")
	}
	if !mat.Equals(input.Data(), result.Data()) {
		t.Fatal("result values changed")
	}
}

func Test_Layer_activate_sigmoidCorrectlyModifiesTheInput(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionSigmoid)
	input := generateInput(someInputSize, someInputSize)

	result := l1.activate(input)

	expected := input.Sigmoid()
	if !mat.Equals(expected.Data(), result.Data()) {
		t.Fatal("result values changed")
	}
}

func Test_Layer_activate_reluCorrectlyModifiesTheInput(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionRelu)
	input := generateInput(someInputSize, someInputSize)

	result := l1.activate(input)

	expected := input.Relu()
	if !mat.Equals(expected.Data(), result.Data()) {
		t.Fatal("result values changed")
	}
}

func Test_Layer_Forward_singleLayerWithIdentityShouldProduceADotOfInputAndWeightsPlusBias(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l2 := NewOutputLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l1.ConnectTo(l2)
	input := generateInput(someInputSize, someInputSize)

	pred := l1.Forward(input)

	expected := tensor.Add(tensor.Dot(input, l1.weights), tensor.Expand(l1.bias, 0, someInputSize))
	if !mat.Equals(expected.Data(), pred.Data()) {
		t.Fatal("operations are incorrect")
	}
}

func Test_Layer_Forward_singleLayerWithSigmoidShouldProduceADotOfActivatedInputAndWeightsPlusBias(t *testing.T) {
	l1 := NewLayer(someInputSize, tensor.ActivationFunctionSigmoid)
	l2 := NewOutputLayer(someInputSize, tensor.ActivationFunctionIdentity)
	l1.ConnectTo(l2)
	input := generateInput(someInputSize, someInputSize)

	pred := l1.Forward(input)

	expected := tensor.Add(tensor.Dot(input.Sigmoid(), l1.weights), tensor.Expand(l1.bias, 0, someInputSize))
	if !mat.Equals(expected.Data(), pred.Data()) {
		t.Fatal("operations are incorrect")
	}
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
	if !mat.Equals(expected.Data(), pred.Data()) {
		t.Fatal("operations are incorrect")
	}
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