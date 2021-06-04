fluent(holding(P, O)) :- person(P), object(O).

happensAt(close(P, O, Threshold), Time) :-
    person(P), object(O),
    bbox(Time, P,_,_,_,_),
    bbox(Time, O,_,_,_,_),
    bbox_intersec(Time, O, P, Intersection),
    holding_threshold(Threshold),
    Intersection >= Threshold.

holdsAt(F,T + 1) :-
    fluent(F), time(T), time(T + 1),
    initiatedAt(F,T), not terminatedAt(F, T).

holdsAt(F,T + 1) :-
    fluent(F), time(T), time(T + 1),
    holdsAt(F,T), not terminatedAt(F,T).

time(0..200).

holding_threshold(1..100).

:- current_time(T-1), not holdsAt(A, T), goal(holdsAt(A,T)).
:- current_time(T-1), holdsAt(A, T), not goal(holdsAt(A,T)).

#modeh(initiatedAt(holding(var(person), var(object)), var(time))).
#modeb(1, holdsAt(holding(var(person), var(object)), var(time))).
#modeb(1, happensAt(close(var(person), var(object), const(holding_threshold)), var(time))).
#constant(holding_threshold, 1..100).
