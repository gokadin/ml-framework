package tensor

import (
	"fmt"
	"strings"
)

type op interface {
	name() string
	dependencies() []*Tensor
	forwardShape() Shape
	backwardShapes(tensorShape Shape) []Shape
	forward(tensor *Tensor)
	backward(tensor *Tensor)
}

func handleIncompatibleShapes(opName string, shapes ...Shape) {
	shapesStrings := make([]string, len(shapes))
	for i := 0; i < len(shapes); i++ {
		shapesStrings[i] = fmt.Sprintf("%dx%d", shapes[i].X, shapes[i].Y)
		strings.Join(shapesStrings, " - ")
	}

	panic(fmt.Sprintf("incompatible shapes for operation %s: %s", opName, shapesStrings))
}
