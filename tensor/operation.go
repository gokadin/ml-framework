package tensor

import (
	"log"
)

const (
	operationNone = "operationNone"
	operationAdd = "operationAdd"
	operationSub = "operationSub"
	operationPow = "operationPow"
	operationDivScalar = "operationDivScalar"
	operationSum = "operationSum"
	operationExpand = "operationExpand"
	operationDot = "operationDot"
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

func (o *operation) differentiate(grad [][]float64) [][]float64 {
    switch o.name {
	case operationNone:
        return o.differentiateNone(grad)
	case operationAdd:
        return o.differentiateAdd(grad)
	case operationSub:
		return o.differentiateSub(grad)
	case operationPow:
		return o.differentiatePow(grad)
	case operationDivScalar:
		return o.differentiateDivScalar(grad)
	case operationSum:
		return o.differentiateSum(grad)
	case operationExpand:
		return o.differentiateExpand(grad)
	case operationDot:
		return o.differentiateDot(grad)
	default:
		log.Fatalf("differentiation not supported")
		return nil
	}
}

func (o *operation) differentiateNone(grad [][]float64) [][]float64 {
    return grad
}

func (o *operation) differentiateAdd(grad [][]float64) [][]float64 {
	return grad
}

func (o *operation) differentiateSub(grad [][]float64) [][]float64 {
	if o.children[0].isMarked {
		return grad
	}
	if o.children[1].isMarked {
		return mulScalar(grad, -1)
	}
	log.Fatal("could not find a marked child")
	return nil
}

func (o *operation) differentiatePow(grad [][]float64) [][]float64 {
	if o.metadata[0] == 2 {
		return mul(grad, mulScalar(o.children[0].tensor.mat, 2))
	}
	return mul(grad, mulScalar(pow(o.children[0].tensor.mat, o.metadata[0] - 1), o.metadata[0]))
}

func (o *operation) differentiateDivScalar(grad [][]float64) [][]float64 {
	return mulScalar(grad, 1 / o.metadata[0])
}

func (o *operation) differentiateSum(grad [][]float64) [][]float64 {
    if o.metadata[0] == 0 {
    	return o.differentiateSumX(grad)
	}

    log.Fatal("sum y derivative not yet supported")
	return nil
}

func (o *operation) differentiateSumX(grad [][]float64) [][]float64 {
	return expand(grad, 0, len(o.children[0].tensor.mat))
}

func (o *operation) differentiateExpand(grad [][]float64) [][]float64 {
	if o.metadata[0] == 0 {
        return o.differentiateExpandX(grad)
	}

	log.Fatal("expand y derivative not yet supported")
	return nil
}

func (o *operation) differentiateExpandX(grad [][]float64) [][]float64 {
    return sum(grad, 0)
}

func (o *operation) differentiateDot(grad [][]float64) [][]float64 {
	if o.children[0].isMarked {
        return dot(grad, transpose(o.children[1].tensor.mat))
	}
	if o.children[1].isMarked {
		return transpose(dot(transpose(grad), o.children[0].tensor.mat))
	}
	log.Fatal("could not find a marked child")
	return nil
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
