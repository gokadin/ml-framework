package core

import (
	"github.com/gokadin/ml-framework/layer"
	"github.com/gokadin/ml-framework/tensor"
	"log"
)

type builder struct {
	network *Network
}

func newBuilder(network *Network) *builder {
	return &builder{
		network: network,
	}
}

func (b *builder) addInputLayer(size int) {
	if len(b.network.layers) != 0 {
		log.Fatal("You cannot add an input layer after adding other layers.")
	}

	b.network.layers = append(b.network.layers, layer.NewLayer(size, tensor.ActivationFunctionIdentity))
}

func (b *builder) AddHiddenLayer(size int, activationFunctionName string) *builder {
	if len(b.network.layers) == 0 {
		log.Fatal("You must add an input layer before adding a hidden layer.")
	}

	b.addLayer(layer.NewLayer(size, activationFunctionName))
	return b
}

func (b *builder) AddOutputLayer(size int, activationFunctionName string) *Network {
	if len(b.network.layers) < 1 {
		log.Fatal("You must add an input layer before adding an output layer.")
	}

	b.addLayer(layer.NewOutputLayer(size, activationFunctionName))
	return b.network
}

func (b *builder) addLayer(l *layer.Layer) {
	b.network.layers[len(b.network.layers) - 1].ConnectTo(l)
	b.network.layers = append(b.network.layers, l)
}
