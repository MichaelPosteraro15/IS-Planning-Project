(define (problem adam)
  (:domain snowman_numeric)

  ;; --------------------------------------------------------------------
  ;; OGGETTI
  ;; --------------------------------------------------------------------
  (:objects
    right left up down                   - direction
    ball_0 ball_1 ball_2                 - ball
    loc_1_1 loc_1_2 loc_1_3 loc_1_4 loc_1_5
    loc_2_1 loc_2_3 loc_2_5
    loc_3_1 loc_3_2 loc_3_3 loc_3_4 loc_3_5
    loc_4_1 loc_4_3 loc_4_5
    loc_5_1 loc_5_2 loc_5_3 loc_5_4 loc_5_5  - location
  ) 

  ;; --------------------------------------------------------------------
  ;; STATO INIZIALE
  ;; --------------------------------------------------------------------
  (:init
    ;; costo iniziale
    (= (total-cost) 0)

    ;; ---------- grafo di adiacenza ------------------------------------
    (next loc_1_1 loc_2_1 right)   (next loc_1_1 loc_1_2 up)
    (next loc_1_2 loc_1_3 up)      (next loc_1_2 loc_1_1 down)
    (next loc_1_3 loc_2_3 right)   (next loc_1_3 loc_1_4 up) (next loc_1_3 loc_1_2 down)
    (next loc_1_4 loc_1_5 up)      (next loc_1_4 loc_1_3 down)
    (next loc_1_5 loc_2_5 right)   (next loc_1_5 loc_1_4 down)

    (next loc_2_1 loc_3_1 right)   (next loc_2_1 loc_1_1 left)
    (next loc_2_3 loc_3_3 right)   (next loc_2_3 loc_1_3 left)
    (next loc_2_5 loc_3_5 right)   (next loc_2_5 loc_1_5 left)

    (next loc_3_1 loc_4_1 right)   (next loc_3_1 loc_2_1 left)
    (next loc_3_1 loc_3_2 up)
    (next loc_3_2 loc_3_3 up)      (next loc_3_2 loc_3_1 down)
    (next loc_3_3 loc_4_3 right)   (next loc_3_3 loc_2_3 left)
    (next loc_3_3 loc_3_4 up)      (next loc_3_3 loc_3_2 down)
    (next loc_3_4 loc_3_5 up)      (next loc_3_4 loc_3_3 down)
    (next loc_3_5 loc_4_5 right)   (next loc_3_5 loc_2_5 left) (next loc_3_5 loc_3_4 down)

    (next loc_4_1 loc_5_1 right)   (next loc_4_1 loc_3_1 left)
    (next loc_4_3 loc_5_3 right)   (next loc_4_3 loc_3_3 left)
    (next loc_4_5 loc_5_5 right)   (next loc_4_5 loc_3_5 left)

    (next loc_5_1 loc_4_1 left)    (next loc_5_1 loc_5_2 up)
    (next loc_5_2 loc_5_3 up)      (next loc_5_2 loc_5_1 down)
    (next loc_5_3 loc_4_3 left)    (next loc_5_3 loc_5_4 up) (next loc_5_3 loc_5_2 down)
    (next loc_5_4 loc_5_5 up)      (next loc_5_4 loc_5_3 down)
    (next loc_5_5 loc_4_5 left)    (next loc_5_5 loc_5_4 down)

    ;; ---------- tipi di terreno --------------------------------------
    ;; 1 = neve, 0 = vuoto
    (= (location_type loc_1_1) 1)
    (= (location_type loc_1_2) 1)
    (= (location_type loc_1_3) 1)
    (= (location_type loc_1_4) 1)
    (= (location_type loc_1_5) 1)

    (= (location_type loc_2_1) 1)
    (= (location_type loc_2_3) 0)
    (= (location_type loc_2_5) 1)

    (= (location_type loc_3_1) 1)
    (= (location_type loc_3_2) 0)
    (= (location_type loc_3_3) 0)
    (= (location_type loc_3_4) 0)
    (= (location_type loc_3_5) 1)

    (= (location_type loc_4_1) 1)
    (= (location_type loc_4_3) 0)
    (= (location_type loc_4_5) 1)

    (= (location_type loc_5_1) 1)
    (= (location_type loc_5_2) 1)
    (= (location_type loc_5_3) 1)
    (= (location_type loc_5_4) 1)
    (= (location_type loc_5_5) 1)


    ;; ---------- palle ---------------------------------------------------
    (ball_at ball_0 loc_2_3)   (= (ball_size ball_0) 0)
    (ball_at ball_1 loc_3_3)   (= (ball_size ball_1) 0)
    (ball_at ball_2 loc_4_3)   (= (ball_size ball_2) 0)

    ;; ---------- personaggio --------------------------------------------
    (character_at loc_3_2)
  )

  ;; --------------------------------------------------------------------
  ;; OBIETTIVO
  ;; --------------------------------------------------------------------
  (:goal (goal))
  

  ;; --------------------------------------------------------------------
  ;; METRICA
  ;; --------------------------------------------------------------------
  (:metric minimize (total-cost))
)
