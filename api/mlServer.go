package api

import (
	"encoding/json"
	"fmt"
	"github.com/gorilla/mux"
	"log"
	"ml-framework/persistence"
	"net/http"
)

type MLServer struct {
	upgrader websocket.Upgrader
}

func NewMLServer() *MLServer {
	return &MLServer{
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
		},
	}
}

func (s *MLServer) Start() {
	router := mux.NewRouter()

	router.HandleFunc("/echo", func(w http.ResponseWriter, r *http.Request) {
		s.upgrader.CheckOrigin = func(r *http.Request) bool { return true }
		conn, err := s.upgrader.Upgrade(w, r, nil)
		if err != nil {
			fmt.Println(err)
			return
		}

		for {
			msgType, message, err := conn.ReadMessage()
			if err != nil {
				fmt.Println(err)
				return
			}

			fmt.Println(fmt.Sprintf("%s sent: %s", conn.RemoteAddr(), string(message)))

			if err = conn.WriteMessage(msgType, message); err != nil {
				fmt.Println(err)
				return
			}
		}
	})

	router.HandleFunc("/projects", getProjects).Methods("GET", "OPTIONS")
	router.HandleFunc("/describe", describe).Methods("GET", "OPTIONS")
	log.Fatal(http.ListenAndServe(":3004", router))
}

func getProjects(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	_ = json.NewEncoder(w).Encode([]*project{
		{Id: 1, Name: "p1"},
		{Id: 2, Name: "p2"},
	})
}

type project struct {
	Id   int    `json:"id"`
	Name string `json:"name"`
}

func describe(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	_ = json.NewEncoder(w).Encode(persistence.BuildDefinition("experimental-rl-batch"))
}
