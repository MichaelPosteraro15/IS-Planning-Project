(define (domain snowman_basic_adl)

  ;; Domain requirements
  (:requirements
    :typing
    :negative-preconditions
    :equality
    :disjunctive-preconditions
    :conditional-effects
    :action-costs
  )

  ;; Objects types
  (:types
    location direction ball size - object
  )

  ;; Predicates
  (:predicates
    (snow ?l - location)
    (next ?from ?to - location ?dir - direction)
    (occupancy ?l - location)
    (character_at ?l - location)
    (ball_at ?b - ball ?l - location)
    (ball_size_small ?b - ball)
    (ball_size_medium ?b - ball)
    (ball_size_large ?b - ball)
    (goal)
  )

  ;; Numeric functions
  (:functions
    (total-cost) - number
  )

  ;; Action for moving the character
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

  ;; Action for moving the ball
  ;; The parameters are:
  ;; ?b (ball) ?from (initial ball location) and ?to (target ball location)
  ;; ?ppos (initial character location)
  ;; ?d (movement direction)
  (:action move_ball
    :parameters (?b - ball ?ppos ?from ?to - location ?dir - direction)
    :precondition
      (and
        ;; 1. Initial character position ?ppos and initial ball position ?from are next to each other in ?d direction
        (next ?ppos ?from ?dir)
        ;; 2. Initial ball position ?from and target ball position ?to are next to each other in ?d direction
        (next ?from ?to ?dir)
        ;; 3. Ball ?b must be in Initial ball poition ?from
        (ball_at ?b ?from)
        ;; 4. Character must be in initial character position ?ppos
        (character_at ?ppos)

        ;; 5. This condition is useful to check if I can move a ball already stacked.
        (forall (?o - ball)
          (or
            ;; 5a. The ball ?b and ?o are the same
            (= ?o ?b)
            (or
              ;; 5b. The ball ?b and ?o are different but not in the same location
              (not (ball_at ?o ?from))
              ;; 5c. The ball ?b and ?o are in the same position but ?b is always smaller than ?o
              ;; This is done to ensure stacking rules
              (or
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