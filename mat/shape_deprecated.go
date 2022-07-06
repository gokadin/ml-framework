package mat

type Shape struct {
	X, Y int
}

func WithShape(x, y int) Shape {
	return Shape{x, y}
}
