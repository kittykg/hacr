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

next_time(T, T+1) :- time(T).

holding_threshold(1..100).

#show holdsAt/2.
