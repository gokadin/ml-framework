package tensor

func (t *Tensor) addOperation(operation *operation) {
	t.operations = append(t.operations, operation)
}

func (t *Tensor) Backward() {
	t.setInitialGradient()
	t.backpropagate()
}

func (t *Tensor) backpropagate() {
	for _, operation := range t.operations {
		operation.backward()
		for _, creator := range operation.creators {
			if !creator.isLeaf() {
                creator.backpropagate()
			}
		}
	}
}

func (t *Tensor) isLeaf() bool {
	return len(t.operations) == 0
}

func (t *Tensor) setInitialGradient() {
	t.grad = make([][]float64, len(t.mat))
	for i := range t.grad {
		t.grad[i] = make([]float64, len(t.mat[i]))
		for j := range t.grad[i] {
			t.grad[i][j] = 1.0
		}
	}
}
