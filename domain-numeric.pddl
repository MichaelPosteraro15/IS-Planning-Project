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
    
    ;; For type location (0=empty, 1=snow)
    (location_type ?l - location)
  )
  
  ;; Action for moving the character
  (:action move_character
    :parameters (?from ?to - location ?dir - direction)
    :precondition
      (and
        (next ?from ?to ?dir)
        (character_at ?from)
        ;; The character can only step on snow and empty location. He can't clib balls or stack of balls
        ;; Check directly that no ball is present at the destination
        (forall (?b - ball) (not (ball_at ?b ?to)))
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
         (next ?from ?to  ?dir)
         ;; 3. Ball ?b must be in Initial ball poition ?from
         (ball_at     ?b   ?from)
         ;; 4. Character must be in initial character position ?ppos
         (character_at      ?ppos)

         ;; 5. For each ball ?o, if the ball is in ?from position and it isn't the ?b ball then, it must be bigger then ?b
         ;; with imply it means "IF a ball ?o different from ?b in ?from position THEN it must be smaller then ?b"
         (forall (?o - ball)
           (imply (and (ball_at ?o ?from) (not (= ?o ?b)))
                  (>= (ball_size ?o) (ball_size ?b)))
         )
         
         ;; 6. This condition guarantee that if I want to move the ball at least one of the following must be true:
         ;; the target cell must be free
         ;; the starting cell must be free
         ;; We this we forbid the ball movement from one stack to another (that's the rule)
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

         ;; 7. This condition is useful to check if I can move a ball on a ball stack. In particular, the ball ?b must be smaller
         (forall (?o - ball)
           (imply (ball_at ?o ?to)
                  (> (ball_size ?o) (ball_size ?b)))
         )
       )
     :effect
       (and
         ;; No more ball ?b in ?from position
         (not (ball_at ?b ?from))
         ;; Ball ?b is now in ?to position
         (ball_at ?b ?to)
         ;; Conditionally move the character forward if the “from” cell is now empty
         (when
           (not (exists (?o - ball)
                (and (not (= ?o ?b)) (ball_at ?o ?from))))
           (and
             (not (character_at ?ppos))
             (character_at ?from)
           )
         )

         ;; 3a. If the ball rolls on snow then it must grown in size if it doesn't have the maximum size
         (when (and (= (location_type ?to) 1)
                    (< (ball_size ?b) 2))
               (increase (ball_size ?b) 1))

         ;; 3b. Snow has been removed
         (when (= (location_type ?to) 1)
               (assign (location_type ?to) 0))

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