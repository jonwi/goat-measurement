package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
)

type Payload struct {
	Angle          float32
	Distance       float32
	Image          string
	BodyLength     float32
	ShoulderHeight float32
	RumpHeight     float32
	Weight         float32
}

func handler(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Println("could not read body of request")
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	log.Println(string(body))
	var p Payload
	err = json.Unmarshal(body, &p)
	if err != nil {
		log.Println("could not unmarshal request")
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	f, err := os.OpenFile("output.json", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0777)
	if err != nil {
		log.Println("could not write data")
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	defer f.Close()
	body = append(body, []byte("\n")...)
	_, err = f.Write(body)
	if err != nil {
		log.Println("could not write data")
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	fmt.Println("handled")
	w.WriteHeader(http.StatusOK)
	return
}

func enableCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// TODO: Production
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func main() {
	http.Handle("/", enableCORS(http.HandlerFunc(handler)))
	log.Fatal(http.ListenAndServe(":8080", nil))
}
