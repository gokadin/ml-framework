package graphics2D

import (
	"fmt"
	"github.com/gokadin/ml-framework/mat"
)

type GridWorld struct {
	width int
	height int
	gridSize int
	gridLengthPx float64
	gridHeightPx float64
	playerGrid [][]int
	wallGrid [][]int
	targetGrid [][]int
	dangerGrid [][]int
	rewardGrid [][]int
	agentPositionI int
	agentPositionJ int
}

func NewGridWorld(width, height, gridSize int) *GridWorld {
	w := &GridWorld{
		width: width,
		height: height,
		gridSize: gridSize,
		gridLengthPx: float64(width) / float64(gridSize),
		gridHeightPx: float64(height) / float64(gridSize),
	}

	w.rewardGrid = w.makeGrid(gridSize, -1)

	return w
}

func (gw *GridWorld) makeGrid(size, defaultValue int) [][]int {
	grid := make([][]int, size)
	for i := 0; i < size; i++ {
		row := make([]int, size)
		for j := 0; j < size; j++ {
			row[j] = defaultValue
		}
		grid[i] = row
	}
	return grid
}

func (gw *GridWorld) PlaceAgent(posI, posJ int) {
	gw.playerGrid = gw.makeGrid(gw.gridSize, 0)
	gw.playerGrid[posI][posJ] = 1
	gw.agentPositionI = posI
	gw.agentPositionJ = posJ
}

func (gw *GridWorld) PlaceWall(posI, posY int) {
	gw.wallGrid = gw.makeGrid(gw.gridSize, 0)
	gw.wallGrid[posI][posY] = 1
}

func (gw *GridWorld) PlaceDanger(posI, posY int) {
	gw.dangerGrid = gw.makeGrid(gw.gridSize, 0)
	gw.dangerGrid[posI][posY] = 1
	gw.rewardGrid[posI][posY] = -10
}

func (gw *GridWorld) PlaceTarget(posI, posY int) {
	gw.targetGrid = gw.makeGrid(gw.gridSize, 0)
	gw.targetGrid[posI][posY] = 1
	gw.rewardGrid[posI][posY] = 10
}

func (gw *GridWorld) Print() {
	for i := 0; i < gw.gridSize; i++ {
		for j := 0; j < gw.gridSize; j++ {
			J := (gw.gridSize - 1) - i
			fmt.Print(" | ")
			character := " "
			if gw.playerGrid[j][J] == 1 {
				character = "P"
			}
			if gw.targetGrid[j][J] == 1 {
				character = "+"
			}
			if gw.dangerGrid[j][J] == 1 {
				character = "-"
			}
			if gw.wallGrid[j][J] == 1 {
				character = "w"
			}
			fmt.Print(character)
		}
		fmt.Println(" |")
	}
}

func (gw *GridWorld) PrintState(state *mat.Mat32f) {
	x := state.Equals32f(mat.NewMat32f(mat.WithShape(1, 64), []float32{
		1, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,

		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 1, 0,

		0, 0, 0, 0,
		0, 0, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 0,

		0, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}))

	if !x {
		fmt.Println("NOT EQ")
		fmt.Print(state)
	} else {
		fmt.Print("equals")
	}
}

func (gw *GridWorld) GetState() *mat.Mat32f {
	state := mat.NewMat32fZeros(mat.WithShape(1, gw.gridSize * gw.gridSize * 4))
	for i := 0; i < gw.gridSize; i++ {
		for j := 0; j < gw.gridSize; j++ {
			J := (gw.gridSize - 1) - i
			if gw.playerGrid[j][J] == 1 {
				state.Set(i * gw.gridSize + j, 1)
			}
			if gw.targetGrid[j][J] == 1 {
				state.Set(i * gw.gridSize + j + (gw.gridSize * gw.gridSize), 1)
			}
			if gw.dangerGrid[j][J] == 1 {
				state.Set(i * gw.gridSize + j + (gw.gridSize * gw.gridSize * 2), 1)
			}
			if gw.wallGrid[j][J] == 1 {
				state.Set(i * gw.gridSize + j + (gw.gridSize * gw.gridSize * 3), 1)
			}
		}
	}
	return state
}

func (gw *GridWorld) GetReward() int {
	return gw.rewardGrid[gw.agentPositionI][gw.agentPositionJ]
}

func (gw *GridWorld) MakeMove(action int) {
	var newI int
	var newJ int
	switch action {
	case 2: // down
		if gw.agentPositionJ == 0 {
			return
		}
		newI = gw.agentPositionI
		newJ = gw.agentPositionJ - 1
		break
	case 1: // right
		if gw.agentPositionI == gw.gridSize - 1 {
			return
		}
		newI = gw.agentPositionI + 1
		newJ = gw.agentPositionJ
		break
	case 0: // up
		if gw.agentPositionJ == gw.gridSize - 1 {
			return
		}
		newI = gw.agentPositionI
		newJ = gw.agentPositionJ + 1
		break
	case 3: // left
		if gw.agentPositionI == 0 {
			return
		}
		newI = gw.agentPositionI - 1
		newJ = gw.agentPositionJ
		break
	}
	if gw.wallGrid[newI][newJ] == 1 {
		return
	}
	gw.PlaceAgent(newI, newJ)
}
