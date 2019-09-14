package layer

import (
	"github.com/gokadin/ml-framework/tensor"
	"log"
	"math/rand"
)

type Layer struct {
	activationFunctionName string
	isOutputLayer bool
	weights *tensor.Tensor
    bias *tensor.Tensor
	inputSize int
	nextLayer *Layer
}

func NewLayer(inputSize int, activationFunctionName string) *Layer {
	return newLayer(inputSize, activationFunctionName, false)
}

func NewOutputLayer(inputSize int, activationFunctionName string) *Layer {
	return newLayer(inputSize, activationFunctionName, true)
}

func newLayer(inputSize int, activationFunctionName string, isOutputLayer bool) *Layer {
	return &Layer{
		activationFunctionName: 	  activationFunctionName,
		isOutputLayer:                isOutputLayer,
		inputSize: inputSize,
	}
}

func (l *Layer) ConnectTo(nextLayer *Layer) {
    l.nextLayer = nextLayer
    l.initializeWeightsAndBias()
}

func (l *Layer) initializeWeightsAndBias() {
    weightsMat := make([][]float64, l.inputSize)
    biasMat := make([][]float64, 1)
    biasMat[0] = make([]float64, l.nextLayer.inputSize)
    for i := range weightsMat {
    	weightsMat[i] = make([]float64, l.nextLayer.inputSize)
    	for j := range weightsMat[i] {
    		if i == 0 {
    			biasMat[0][j] = 0
			}
    		weightsMat[i][j] = rand.Float64()
		}
	}
    l.weights = tensor.NewTensor(weightsMat)
    l.bias = tensor.NewTensor(biasMat)
}

func (l *Layer) Forward(input *tensor.Tensor) *tensor.Tensor {
	pred := tensor.Dot(l.activate(input), l.weights)
	//pred := l.activate(tensor.Add(tensor.Dot(input, l.weights), tensor.ExpandX(l.bias, len(input.Data()))))
	if l.nextLayer != nil && !l.nextLayer.isOutputLayer {
		return l.nextLayer.Forward(pred)
	}
	return pred
}

func (l *Layer) activate(in *tensor.Tensor) *tensor.Tensor {
	switch l.activationFunctionName {
	case tensor.ActivationFunctionIdentity:
		return in
	case tensor.ActivationFunctionSigmoid:
		return in.Sigmoid()
	default:
		log.Fatal("activation function is unknown:", l.activationFunctionName)
		return nil
	}
}

func (l *Layer) IsOutputLayer() bool {
	return l.isOutputLayer
}

func (l *Layer) GetParameters() []*tensor.Tensor {
    return []*tensor.Tensor{l.weights}
	//return []*tensor.Tensor{l.weights, l.bias}
}