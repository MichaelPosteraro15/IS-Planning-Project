(define (domain snowman_numeric)

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
    location direction ball - object
  )

  ;; Predicates
  ;; Keep the location related predicates because we can't create numeric function that depends on two different variables
  ;; Furthermore it is more convenient to keep using location
  (:predicates
    (next ?from ?to - location ?dir - direction)
    (character_at ?l - location)
    (ball_at ?b - ball ?l - location)
    
    (goal)
  )

  ;; Numeric functions
  (:functions
    (total-cost) - number
    
    ;; For ball size (0=small, 1=medium, 2=large)
    (ball_size ?b - ball)
    
    ;; For type location (0=empty, 1=snow, 2=occupied by ball/stack of balls)
    (location_type ?l - location)
  )
  
  ;; Action for moving the character
  (:action move_character
    :parameters (?from ?to - location ?dir - direction)
    :precondition
      (and
        (next ?from ?to ?dir)
        (character_at ?from)
        ;; The character can only step on snow and empty location. He can't clib balls and stack of balls
        (not (= (location_type ?to) 2))
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
         
         (next ?ppos ?from ?dir)
         (next ?from ?to  ?dir)
         (ball_at     ?b   ?from)
         (character_at      ?ppos)

         (forall (?o - ball)
           (imply (and (ball_at ?o ?from) (not (= ?o ?b)))
                  (>= (ball_size ?o) (ball_size ?b)))
         )
         
         (or
          (forall (?o - ball)
            (or
              (= ?o ?b)
              (not (ball_at ?o ?from))
            )
          )
          (forall (?o - ball)
            (not (ball_at ?o ?to))
          )
         )

         (forall (?o - ball)
           (imply (ball_at ?o ?to)
                  (> (ball_size ?o) (ball_size ?b)))
         )
       )
     :effect
       (and
         ;; 1. La palla avanza
         (not (ball_at ?b ?from))
         (ball_at ?b ?to)
         
         ;; Location ?to is now occupied
         (assign (location_type ?to) 2)

         ;; 2. Se in ?from non resta nessun’altra palla, il personaggio avanza
         (when
           (not (exists (?o - ball)
                (and (not (= ?o ?b)) (ball_at ?o ?from))))
           (and
             (not (character_at ?ppos))
             (character_at ?from)
           )
         )

         ;; 3a. Se rotola su neve e la palla non è già grande, aumenta di taglia
         (when (and (= (location_type ?to) 1)
                    (< (ball_size ?b) 2))
               (increase (ball_size ?b) 1))

         ;; 3b. La neve viene rimossa
         (when (= (location_type ?to) 1)
               (assign (location_type ?to) 0))

         ;; 4. Costo del movimento
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