package tensor

import (
	"log"
)

const (
	operationNone = "operationNone"
	operationAdd = "operationAdd"
	operationSub = "operationSub"
	operationPow = "operationPow"
)

type operation struct {
	name string
	tensor *Tensor
	gradient [][]float64
	children []*operation
	metadata []float64
	isMarked bool
}

func newOperation(name string, tensor *Tensor, children []*operation, metadata ...float64) *operation {
	return &operation{
		name: name,
		tensor: tensor,
		children: children,
		metadata: metadata,
	}
}

func (o *operation) isLeaf() bool {
	return o.name == operationNone
}

func (o *operation) differentiate() [][]float64 {
    switch o.name {
	case operationNone:
        return generateIdentityGradient(o.tensor.mat)
        //return o.tensor.mat
	case operationAdd:
        return findMarkedChild(o).differentiate()
	case operationSub:
		if o.children[0].isMarked {
			return o.children[0].differentiate()
		}
		if o.children[1].isMarked {
			return mulScalar(o.children[1].differentiate(), -1)
		}
		log.Fatal("could not find a marked child")
		return nil
	case operationPow:
		if o.metadata[0] == 2 {
            return mul(mulScalar(o.children[0].tensor.mat, 2), o.children[0].differentiate())
		}
		return mul(mulScalar(pow(o.children[0].tensor.mat, o.metadata[0] - 1), o.metadata[0]), o.children[0].differentiate())
	default:
		log.Fatalf("differentiation not supported")
		return nil
	}
}

func generateIdentityGradient(mat [][]float64) [][]float64 {
	grad := make([][]float64, len(mat))
	for i := range grad {
		grad[i] = make([]float64, len(mat[i]))
		for j := range grad[i] {
			grad[i][j] = 1
		}
	}
	return grad
}

func findMarkedChild(operation *operation) *operation {
    for _, child := range operation.children {
    	if child.isMarked {
    		return child
		}
	}
    
    log.Fatalf("could not find a marked child")
    return nil
}
