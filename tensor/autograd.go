package tensor

import "log"

type Autograd struct {
	cache map[string][]*operation
}

func NewAutograd() *Autograd {
	return &Autograd{
		cache: make(map[string][]*operation),
	}
}

func (a *Autograd) Derivative(c, f *Tensor) [][]float64 {
	return computeDerivative(f.operation, a.getDerivativeGraph(f))
}

func (a *Autograd) Backward(t *Tensor) {
    a.backwardRecursive(t, generateIdentityGradient(t.mat))
}

func (a *Autograd) backwardRecursive(t *Tensor, grad [][]float64) {
	t.operation.differentiate(grad)
	for _, child := range t.operation.children {
        a.backwardRecursive(child.tensor, t.operation.gradient)
	}
}

func (a *Autograd) getDerivativeGraph(t *Tensor) []*operation {
	if _, ok := a.cache[t.id]; ok {
		return a.cache[t.id]
	}
	
    a.cache[t.id] = createDerivativeGraph(t.operation)
    return a.cache[t.id]
}

func calculateDerivativeHash(c, f *Tensor) string {
	return c.id + f.id
}

func createDerivativeGraph(f *operation) []*operation {
    graph := make([]*operation, 0)
	graph = append(graph, f)

    current := f
	for !current.isLeaf() {
		//current = findThatSpecialChild(c, current.children)
		current.isMarked = true
		graph = append(graph, current)
	}

    return optimizeGraph(graph)
}

func optimizeGraph(graph []*operation) []*operation {
	optimized := make([]*operation, 0)
    for _, operation := range graph {
    	if operation.name == operationNone || operation.name == operationAdd {
            continue
		}
    	if operation.name == operationSub && operation.children[0].isMarked {
    		continue
		}
    	optimized = append(optimized, operation)
	}
    return optimized
}

func findThatSpecialChild(c *operation, children []*operation) *operation {
    for _, child := range children {
    	if child.tensor.id == c.tensor.id {
    		return child
		}
        if leafIsInPath(c, child) {
        	return child
		}
	}
    
    log.Fatal("no child was special")
    return nil
}

func leafIsInPath(c, root *operation) bool {
    if root.tensor.id == c.tensor.id {
    	return true
	}

    for _, child := range root.children {
		if leafIsInPath(c, child) {
			return true
		}
	}

    return false
}

func computeDerivative(root *operation, graph []*operation) [][]float64 {
	previousGrad := generateIdentityGradient(root.tensor.mat)
	for _, operation := range graph {
        operation.differentiate(previousGrad)
        previousGrad = operation.gradient
	}
    return previousGrad
}

