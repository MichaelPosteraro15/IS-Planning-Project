(define (domain snowman_basic_adl)

  ;; Requisiti del dominio
  (:requirements
    :typing
    :negative-preconditions
    :equality
    :disjunctive-preconditions
    :conditional-effects
    :action-costs
    :fluents
  )

  ;; Tipi di oggetti
  (:types
    block location ball - object
  )

  ;; Predicati
  (:predicates
    (block_location ?block ?l - location)
    (next ?from ?to - location ?dir - direction)
    (character_at ?l - location)
    (ball_at ?b - ball ?l - location)
    (goal)
  )

  ;; Funzioni numeriche
  (:functions
    (total-cost) - number

    ;; 0 blocco normale
    ;; 1 blocco con la neve
    ;; 3 blocco occupato
    (block_type ?b - block)
  
    ;; 0 palla piccola
    ;; 1 palla media
    ;; 2 palla grande
    (ball_size ?b - ball)
  )

  ;; Azione: muovi il personaggio
  (:action move_character
    :parameters (?from ?to - location ?dir - direction)
    :precondition
      (and
        (next ?from ?to ?dir)
        (character_at ?from)
        (not (occupancy ?to))
      )
    :effect
      (and
        (not (character_at ?from))
        (character_at ?to)
      )
  )

  ;; Azione: muovi una palla
  (:action move_ball
    :parameters (?b - ball ?ppos ?from ?to - location ?dir - direction)
    :precondition
      (and
        ;; Il personaggio è dietro la palla e la palla può avanzare
        (next ?ppos ?from ?dir)
        (next ?from ?to ?dir)

        ;; Presenza della pall in ?from e del personaggio in ?ppos dietro la palla
        (ball_at ?b ?from)
        (character_at ?ppos)

        ;; Per ognuna delle palle del dominio deve valere almeno una delle seguenti condizioni.
        ;; Se ciò accade allora la precondizione passa e quindi si arriva all'effecy.
        (forall (?o - ball)
          (or
            ;; 1) Se si verifica che la palla che voglio spostare ?b è uguale a quella presa in
            ;; considerazione ?o allora la condizione è vera quindi passo all'effect
            (= ?o ?b)
            ;; altrimenti
            (or
              ;; 2) Se la palla che voglio spostare ?b è diversa da quella presa in considerazione
              ;; ?o e la palla ?o non sta nella posizione di ?b che é ?from 
              ;; allora la condizione è vera perchè non ci sono palle ad ostacolare lo spostamento di ?b
              (not (ball_at ?o ?from))
              ;; altrimenti
              (or
                ;; 3) Se la palla che voglio spostare ?b sta nella stessa cella di ?o allora deve valere che 
                ;; ?o è più grande di ?b (la palla ?b che sta sopra ?o deve essere più piccola)
                (and (ball_size_small ?b) (ball_size_medium ?o))
                (and (ball_size_small ?b) (ball_size_large ?o))
                (and (ball_size_medium ?b) (ball_size_large ?o))
              )
            )
          )
        )

        ;; Deve necessariamente valere una di queste condizioni
        (or
          ;; Per ogni palla ?o deve valere almeno una di queste condizioni
          (forall (?o - ball)
            (or
              ;; 1) La palla ?b che voglio spostare è diversa dalla palla presa in consideraione ?o
              (= ?o ?b)
              ;; 2) La palla che voglio spostare ?b non stia nella stessa posizione della palla ?o (diversa da ?b)
              (not (ball_at ?o ?from))
            )
          )
          ;; Per ogni palla ?o non deve valere che questa occupi la posizione finale in cui voglio spostare ?b
          (forall (?o - ball)
            (not (ball_at ?o ?to))
          )
        )

        ;; La palla ?b può andare su un'altra palla solo se è più piccola
        (forall (?o - ball)
          (or
            (not (ball_at ?o ?to))
            (or
              (and (ball_size_small ?b) (ball_size_medium ?o))
              (and (ball_size_small ?b) (ball_size_large ?o))
              (and (ball_size_medium ?b) (ball_size_large ?o))
            )
          )
        )
      )

    :effect
      (and
        (occupancy ?to)
        (not (ball_at ?b ?from))
        (ball_at ?b ?to)

        ;; Se nessun'altra palla è nella locazione precedente, il personaggio si sposta
        (when
          (forall (?o - ball)
            (or
              (= ?o ?b)
              (not (ball_at ?o ?from))
            )
          )
          (and
            (not (character_at ?ppos))
            (character_at ?from)
            (not (occupancy ?from))
          )
        )

        ;; Rimozione della neve
        (not (snow ?to))

        ;; La palla cresce se rotola su neve
        (when
          (and (snow ?to) (ball_size_small ?b))
          (and
            (not (ball_size_small ?b))
            (ball_size_medium ?b)
          )
        )
        (when
          (and (snow ?to) (ball_size_medium ?b))
          (and
            (not (ball_size_medium ?b))
            (ball_size_large ?b)
          )
        )

        ;; Costo dell'azione
        (increase (total-cost) 1)
      )
  )

  ;; Azione: verifica il goal (tre palle impilate nella stessa posizione)
  (:action goal
    :parameters (?b0 ?b1 ?b2 - ball ?p0 - location)
    :precondition
      (and
        (not (= ?b0 ?b1))
        (not (= ?b0 ?b2))
        (not (= ?b1 ?b2))
        (ball_at ?b0 ?p0)
        (ball_at ?b1 ?p0)
        (ball_at ?b2 ?p0)
      )
    :effect
      (goal)
  )
)