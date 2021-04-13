fluent(holding(P, O)) :- person(P), object(O).

happensAt(close(P, O, Threshold), Time) :-
    person(P), object(O),
    bbox(Time, P,_,_,_,_),
    bbox(Time, O,_,_,_,_),
    bbox_intersec(Time, O, P, Intersection),
    holding_threshold(Threshold),
    Intersection >= Threshold.

holdsAt(F,T) :-
    fluent(F), time(Ts), time(T),
    initiatedAt(F,Ts), next_time(Ts, T).
holdsAt(F,T) :-
    fluent(F), time(Ts), time(T),
    holdsAt(F,Ts), not terminatedAt(F,Ts), next_time(Ts, T).

next_time(T, T+1) :- time(T).

time(0..200).

holding_threshold(1..100).

:- current_time(T-1), not holdsAt(A, T), goal(holdsAt(A,T)).
:- current_time(T-1), holdsAt(A, T), not goal(holdsAt(A,T)).

#modeh(initiatedAt(holding(var(person), var(object)), var(time))).
#modeb(1, holdsAt(holding(var(person), var(object)), var(time))).
#modeb(1, happensAt(close(var(person), var(object), const(holding_threshold)), var(time))).
#constant(holding_threshold, 1..100).
