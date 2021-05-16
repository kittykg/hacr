fluent(in_scene(P)) :- person(P).
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

#modeh(initiates(enter(var(person)), at_curr_location(var(person)), var(time))).
#modeb(holdsAt(in_scene(var(person)), var(time))).
#modeb(holdsAt(at_curr_location(var(person)), var(time))).
#modeb(abrupt_transition(var(time), var(time)), (anti_reflexive)).
#modeb(next_time(var(time), var(time)), (positive, anti_reflexive)).
#maxv(3).
