#bias("

cost(R, H) :- meta_rule_instance, head(H), possible_meta(ID, H), recall(ID, R).

1 { normal_rule ; disjunctive_rule : allow_disjunction ; choice_rule : not no_aggregates; constraint : not no_constraints; meta_rule_instance ; weak_constraint } 1.

1 { head(H) : possible_head(_, H) } 1 :- normal_rule.
2 { in_head(H) : possible_head(_, H); conditional_disjunct } :- disjunctive_rule.
:- conditional_disjunct, in_head(H), #false : condition(_, H).

MINHL { in_head(H) : possible_aggregate_head(_, H) } MAXHL :- choice_rule, minhl(MINHL), maxhl(MAXHL).

1 { ub(1..MAXHL) } 1 :- choice_rule, maxhl(MAXHL).
1 { lb(0..UB) } 1 :- choice_rule, ub(UB).

:- lb(LB), ub(UB), LB > UB.

{
  head(H) : possible_head(_, H), normal_rule;
  in_head(H) : possible_head(_, H), disjunctive_rule;
  in_head(H) : possible_aggregate_head(_, H), choice_rule;
  body(B) : possible_body(ID, B), not constraint(ID, negative);
  body(naf(B)) : possible_body(ID, B), not constraint(ID, positive);
  condition(C, B) : in_head(B), possible_conditional(ID, C), not constraint(ID, negative);
  condition(naf(C), B) : in_head(B), possible_conditional(ID, C), not constraint(ID, positive);
  condition(C, B) : body(B), possible_conditional(ID, C), not constraint(ID, negative);
  condition(naf(C), B) : body(B), possible_conditional(ID, C), not constraint(ID, positive)
} N :- max_rule_length(N), not weak_constraint, not meta_rule_instance.


{
  body(B) : possible_body(ID, B), not constraint(ID, negative);
  body(naf(B)) : possible_body(ID, B), not constraint(ID, positive);
  condition(C, B) : body(B), possible_conditional(ID, C), not constraint(ID, negative);
  condition(naf(C), B) : body(B), possible_conditional(ID, C), not constraint(ID, positive)
} N :- max_body_literals(N), not weak_constraint, not meta_rule_instance.

{
  body(B) : possible_body(ID, B), not constraint(ID, negative);
  body(naf(B)) : possible_body(ID, B), not constraint(ID, positive)
} R :- possible_body(ID, _), recall(ID, R), R > 0, not weak_constraint, not meta_rule_instance.

{
  condition(C, B) : possible_conditional(ID, C), not constraint(ID, negative);
  condition(naf(C), B) : possible_conditional(ID, C), not constraint(ID, positive)
} :- possible_conditional(ID, _), recall(ID, R), R > 0, body(B).

{
  weak_body(B) : possible_opt(ID, B), not constraint(ID, negative);
  weak_body(naf(B)) : possible_opt(ID, B), not constraint(ID, positive)
} R :- possible_opt(ID, _), recall(ID, R), R > 0, weak_constraint, not meta_rule_instance.

1 {
  weak_body(B) : possible_opt(ID, B), not constraint(ID, negative);
  weak_body(naf(B)) : possible_opt(ID, B), not constraint(ID, positive)
} N :- max_wc_length(N), weak_constraint, not meta_rule_instance.

variable(var__(1..N)) :- maxv(N).
{ var(V, T) : var_type(T), T != any } 1 :- variable(V).

:- variable(V1), variable(V2), V1 < V2,
   in_head(H1), at(H1, I2, V2),
   in_head(H2), at(H2, I2, V2),
   dmhv.

body(V, T) :- var(V, T), occurs_pos(V), strict_types.
body(V, T) :- var(V, T), occurs_non_pos(V), strict_types.
occurs_pos(V) :- occurs_non_pos(V), strict_types.

occurs_non_pos(V) :- body(V, T).
occurs_non_pos(V) :- variable(V), head(H), at(H, _, V); not condition(_, H).
occurs_pos(V) :- variable(V), body(H), at(H, _, V), #false: H = bin_exp(_, _, _); not condition(_, H).
occurs_pos(V) :- variable(V), weak_body(H), at(H, _, V), #false: H = bin_exp(_, _, _); not condition(_, H).
:- occurs_non_pos(V), not occurs_pos(V), not strict_types.
:- in_head(X), body(X).
:- head(X), body(X).
:- in_head(X), body(naf(X)).
:- head(X), body(naf(X)).
:- body(X), body(naf(X)).

occurs_pos_conditional(L, V) :- variable(V), condition(B, L), at(B, _, V).

occurs_conditional(L, V) :- occurs_pos_conditional(L, V).
occurs_conditional(L, V) :- variable(V), condition(naf(B), L), at(B, _, V).
occurs_conditional(L, V) :- variable(V), condition(bin_exp(X, Y, Z), L), at(bin_exp(X, Y, Z), _, V).

:- occurs_conditional(L, V), occurs_conditional(L2, V), L < L2, not occurs_pos(V).
:- occurs_conditional(L, V), not occurs_pos(V), not occurs_conditional(_, V2), variable(V2), V2 < V.
:- body(X), condition(X, _), #false : condition(_, X).
:- condition(X, X).

:- occurs_conditional(L, V), not occurs_pos_conditional(L, V), not occurs_pos(V).

occurs_non_pos(V) :- variable(V), in_head(H), at(H, _, V); not occurs_pos_conditional(H, V).
occurs_non_pos(V) :- variable(V), body(H), at(H, _, V); condition(_, H); not occurs_pos_conditional(H, V).
occurs_non_pos(V) :- variable(V), weak_body(H), at(H, _, V); condition(_, H); not occurs_pos_conditional(H, V).
occurs_non_pos(V) :- variable(V), body(naf(H)), at(H, _, V); not occurs_pos_conditional(naf(H), V).
occurs_non_pos(V) :- variable(V), weak_body(naf(H)), at(H, _, V); not occurs_pos_conditional(naf(H), V).
occurs_non_pos(V) :- variable(V), body(bin_exp(X, Y, Z)), at(bin_exp(X, Y, Z), _, V); not occurs_pos_conditional(bin_exp(X, Y, Z), V).
occurs_non_pos(V) :- variable(V), weak_body(bin_exp(X, Y, Z)), at(bin_exp(X, Y, Z), _, V); not occurs_pos_conditional(bin_exp(X, Y, Z), V).


negative(naf(B)) :- body(naf(B)).

in_any_body(B) :- body(B), not negative(B).
in_any_body(B) :- body(naf(B)).
in_any_body(B) :- weak_body(B), not negative(B).
in_any_body(B) :- weak_body(naf(B)).

:- occurs_pos(V), variable(V2), not occurs_pos(V2), V2 < V.


:- constraint, not body(_).

1 { priority(1..P) } 1 :- maxp(P), weak_constraint.
1 {
    weight(I) : integer_weight(I);
    weight(V) : type_weight(W), var(V, W), occurs_pos(V);
    weight(minus(V)) : type_weight(W), var(V, W), occurs_pos(V)
} 1 :- weak_constraint.

const(P, any, A) :- const(P, T, A).
const(P, T, A) :- meta_predicate(T, P, A).

1 { head(H) : possible_meta(_, H) } 1 :- meta_rule_instance.

anon(V) :- variable(V); 1 #count { BL, I : body(BL), at(BL, I, V); BL, I : body(naf(BL)), at(BL, I, V); BL, I : weak_body(BL), at(BL, I, V); BL, I : weak_body(naf(BL)), at(BL, I, V); BL, I : head(BL), at(BL, I, V); BL, I : in_head(BL), at(BL, I, V) } 1.

#show cost/2.
#show head/1.
#show in_head/1.
#show body/1.
#show body/2.
#show condition/2.
#show weak_body/1.
#show lb/1.
#show ub/1.
#show priority/1.
#show weight/1.
#show meta_rule_instance/0.




").
