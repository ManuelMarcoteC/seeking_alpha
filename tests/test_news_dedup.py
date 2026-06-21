"""Token-set Jaccard dedup of syndicated headlines (no embeddings).

Portado del TIL de SOFA: el trabajo real lo hace el tokenizador (stopwords +
mapa de sinónimos canónicos + sentinela _skip), no el Jaccard.
"""
from qtdata.news import dedup


class TestWords:
    def test_lowercases_strips_punct_and_drops_stopwords(self):
        toks = dedup.words("The drone, a UAV, of Iran near Hormuz")
        assert "the" not in toks and "a" not in toks and "of" not in toks
        assert all(len(t) > 1 for t in toks)

    def test_synonyms_map_to_canonical(self):
        assert "uav" in dedup.words("drones")
        assert "uav" in dedup.words("UAV")
        assert "engage" in dedup.words("intercepts")
        assert "drone" not in dedup.words("drone")  # canonicalizado a uav

    def test_skip_sentinel_drops_filler_verbs(self):
        assert "_skip" not in dedup.words("Apple says revenue up")
        assert "says" not in dedup.words("Apple says revenue up")
        assert "said" not in dedup.words("Tesla said output rose")


class TestSimilarity:
    def test_two_empty_sets_are_identical(self):
        assert dedup.similarity("the a of", "in on at") == 1.0

    def test_one_empty_one_not_is_disjoint(self):
        assert dedup.similarity("the", "Iran drone Hormuz") == 0.0

    def test_syndicated_pair_exceeds_threshold(self):
        a = "Iran intercepts drone near Hormuz"
        b = "Iranian forces engage UAV over Strait of Hormuz"
        assert dedup.similarity(a, b) >= 0.5

    def test_unrelated_sharing_topic_word_stays_below(self):
        a = "Iran intercepts drone near Hormuz"
        b = "Hormuz shipping insurance rates climb sharply"
        assert dedup.similarity(a, b) < 0.5


class TestAssignEventIds:
    def test_collapses_syndicated_titles_into_one_event(self):
        titles = [
            "Iran intercepts drone near Hormuz",
            "Iranian forces engage UAV over Strait of Hormuz",
            "Apple beats Q3 earnings estimates",
        ]
        ids = dedup.assign_event_ids(titles, threshold=0.5)
        assert ids[0] == ids[1]          # las dos del dron -> mismo evento
        assert ids[2] != ids[0]          # Apple -> evento distinto
        assert len(set(ids)) == 2

    def test_singletons_get_unique_ids(self):
        titles = ["Apple beats earnings", "Tesla output rises", "Nvidia upgrade"]
        ids = dedup.assign_event_ids(titles, threshold=0.5)
        assert len(set(ids)) == 3

    def test_empty_input(self):
        assert dedup.assign_event_ids([], threshold=0.5) == []

    def test_single_element(self):
        assert dedup.assign_event_ids(["Apple beats earnings"], threshold=0.5) == [0]

    def test_all_identical_collapse_to_one(self):
        titles = ["Apple beats earnings"] * 4
        ids = dedup.assign_event_ids(titles, threshold=0.5)
        assert len(set(ids)) == 1

    def test_empty_titles_cluster_together_apart_from_real(self):
        # two empty token sets are identical (1.0) -> same event; a real headline
        # stays separate. Pins the documented empty-set semantics against drift.
        ids = dedup.assign_event_ids(["", "", "Apple beats earnings"], threshold=0.5)
        assert ids[0] == ids[1]
        assert ids[2] != ids[0]
