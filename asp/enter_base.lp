fluent(in_camera(P)) :- person(P).
fluent(at_curr_location(P)) :- person(P).

holdsAt(F, T + 1) :-
    time(T), time(T + 1), fluent(F),
    initiates(A, F, T), not clipped(F, T).

holdsAt(F, T + 1) :-
    time(T), time(T + 1), fluent(F),
    holdsAt(F, T), not clipped(F, T).

clipped(F, T) :-
    fluent(F), time(T),
    terminates(A, F, T).

next_time(T1, T1 + 1) :- time(T1), time(T1 + 1).

#show initiates/3.
