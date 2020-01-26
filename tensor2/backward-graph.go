package tensor2

type backwardGraph struct {
	start chan bool
	done chan bool
	root *Tensor
}

func buildBackwardGraph(derivative, of *Tensor) *backwardGraph {
	bg := &backwardGraph{
		start: make(chan bool),
		done: make(chan bool),
		root: of,
	}

	bg.build(derivative, of, true)

	return bg
}

func (bg *backwardGraph) build(derivative, root *Tensor, isFirstRoot bool) (bool, chan bool) {
	if root.op == nil {
		return false, nil
	}

	for _, dependency := range root.op.dependencies() {
		if dependency == nil {
			continue
		}

		if dependency.id == derivative.id {
			dependency.isGradientEnabled = true
			in := make(chan bool)
			go executeBackwardOp(root, in, bg.done)
			return true, in
		}

		if ok, out := bg.build(derivative, dependency, false); ok {
			dependency.isGradientEnabled = true
			if isFirstRoot {
				root.isGradientEnabled = true
				go executeBackwardOp(root, bg.start, out)
				return false, nil
			} else {
				in := make(chan bool)
				go executeBackwardOp(root, in, out)
				return true, in
			}
		}
	}

	return false, nil
}

func (bg *backwardGraph) run() {
	bg.root.grad = generateIdentityGradient(len(bg.root.mat), len(bg.root.mat[0]))
	bg.start <- true
	<-bg.done
}

func executeBackwardOp(tensor *Tensor, in, out chan bool) {
	<-in
	tensor.backward()
	out <- true
}

func generateIdentityGradient(shapeX, shapeY int) [][]float64 {
	grad := make([][]float64, shapeX)
	for i := range grad {
		grad[i] = make([]float64, shapeY)
		for j := range grad[i] {
			grad[i][j] = 1
		}
	}
	return grad
}
