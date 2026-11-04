from barbet_pipeline.full_source_visibility_v2 import tied_observations,stable_scores
from testtime.retrieval import Document
from barbet_pipeline.full_source_visibility_v3 import tied_cutoff_observations


def test_equal_score_order_is_allowed_without_changing_sources():
    a={'id':'a','text':'same source a'};b={'id':'b','text':'same source b'}
    assert tied_observations([a,b],[b,a],{'a':2.0,'b':2.0})
    assert not tied_observations([a,b],[b,a],{'a':2.0,'b':2.000000000000001})
    assert not tied_observations([a,{**b,'text':'corrupted'}],[b,a],{'a':2.0,'b':2.0})
    assert not tied_observations([a,{'id':'c','text':'other'}],[b,a],{'a':2.0,'b':2.0,'c':2.0})
    assert not tied_observations([a,a],[a,b],{'a':2.0,'b':2.0})


def test_stable_scoring_ignores_query_term_order():
    docs=[Document('a','alpha beta alpha','source'),Document('b','beta alpha beta','source')]
    assert stable_scores(docs,'alpha beta')==stable_scores(docs,'beta alpha')
    scores=stable_scores(docs,'alpha beta')
    assert scores['a']==scores['b']


def test_cutoff_tie_requires_exact_sources_and_exact_equal_scores():
    documents={x:{'id':x,'text':'source '+x} for x in ['a','b','c']}
    a,b,c=[documents[x] for x in ['a','b','c']]
    scores={'a':8.0,'b':5.0,'c':5.0}
    assert tied_cutoff_observations([a,c],[a,b],scores,documents)
    assert not tied_cutoff_observations([a,c],[a,b],{**scores,'c':4.999999999999999},documents)
    assert not tied_cutoff_observations([a,{**c,'text':'fake'}],[a,b],scores,documents)
    assert not tied_cutoff_observations([a,a],[a,b],scores,documents)
    assert not tied_cutoff_observations([c,a],[a,b],scores,documents)
