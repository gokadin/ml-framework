package tensor

import (
	"fmt"
	"ml-framework/mat"
	"strings"
)

type op interface {
	name() string
	dependencies() []*Tensor
	forwardShape() mat.Shape
	backwardShapes(tensorShape mat.Shape) []mat.Shape
	forward(tensor *Tensor)
	backward(tensor *Tensor)
}

func handleIncompatibleShapes(opName string, shapes ...mat.Shape) {
	shapesStrings := make([]string, len(shapes))
	for i := 0; i < len(shapes); i++ {
		shapesStrings[i] = fmt.Sprintln(shapes[i].D)
		strings.Join(shapesStrings, " - ")
	}

	panic(fmt.Sprintf("incompatible shapes for operation %s: %s", opName, shapesStrings))
}
