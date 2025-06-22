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

        ;; 5. This condition is useful to check if I can move a ball already stacked. In particular, the ball must be on top
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

        ;; 6. This condition guarantee that if I want to move the ball at least one of the following must be true:
        ;; the target cell must be free
        ;; the starting cell must be free
        ;; We this we forbid the ball movement from one stack to another (that's the rule)
        (or
          ;; To check the emptyness of the initial cell
          (forall (?o - ball)
            (or
              ;; 6a. The ball ?b and ?o are the same
              (= ?o ?b)
              ;; 6b. The ball ?b and ?o are different but not in the same location
              (not (ball_at ?o ?from))
            )
          )
          ;; To check the emptyness of the target cell
          (forall (?o - ball)
            ;; 6c. There must not be a ball in ?to (target cell)
            (not (ball_at ?o ?to))
          ) 
        )

        ;; 7. This condition is useful to check if I can move a ball on a ball stack. In particular, the ball ?b must be smaller
        (forall (?o - ball)
          (or
            ;; 7a. No ball in position ?to (target position)
            (not (ball_at ?o ?to))
            (or
              ;; 7c. The ball ?b and ?o are in the same position but ?b is always smaller than ?o
              ;; This is done to ensure stacking rules
              (and (ball_size_small ?b) (ball_size_medium ?o))
              (and (ball_size_small ?b) (ball_size_large ?o))
              (and (ball_size_medium ?b) (ball_size_large ?o))
            )
          )
        )
      )

    :effect
      (and
        ;; Occupy ?to position
        (occupancy ?to)
        ;; No more ball ?b in ?from position
        (not (ball_at ?b ?from))
        ;; Ball ?b is now in ?to position
        (ball_at ?b ?to)

        ;; Conditionally move the character forward if the “from” cell is now empty
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

        ;; Remove snow in the target cell
        (not (snow ?to))

        ;; If there was snow under the ball, grow the ball one size step
        (when
          (and (snow ?to) (ball_size_small ?b))
          (and
            ;; No longer small
            (not (ball_size_small ?b))
            ;; Now it is medium
            (ball_size_medium ?b)
          )
        )
        (when
          (and (snow ?to) (ball_size_medium ?b))
          (and
            ;; No longer medium
            (not (ball_size_medium ?b))
            ;; Now it becames large
            (ball_size_large ?b)
          )
        )

        ;; Increment the global action cost
        (increase (total-cost) 1)
      )
  )

  ;; To check the final goal
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