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

holding_threshold(1..100).

#show holdsAt/2.
