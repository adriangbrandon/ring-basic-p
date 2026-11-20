/*
 * test-ring-muthu.cpp
 *
 * Characterization test for ring_muthu / c_ring_muthu / ring_sel_muthu (control: ring<>).
 * Compares the number of results of LTJ against a naive BGP evaluator on small in-memory
 * datasets (fixed seed), before and after serializing/loading the index.
 * Exit code != 0 if any check fails.
 *
 * Preconditions of the engine (not tested): IDs >= 1, IDs < 2^32, no variable repeated
 * inside a triple pattern, non-empty dataset.
 */

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <random>
#include <cstdlib>
#include <unistd.h>
#include "ring.hpp"
#include "ring_muthu.hpp"
#include <ltj_algorithm.hpp>
#include <ltj_iterator.hpp>
#include <triple_pattern.hpp>
#include <results_collector.hpp>
#include <veo_adaptive.hpp>
#include <veo_simple.hpp>
#include "utils.hpp"

namespace {

    typedef std::vector<spo_triple> dataset_t;

    struct term {
        bool var;
        uint64_t v;
    };

    struct pattern {
        term s, p, o;
    };

    typedef std::vector<pattern> query_t;

    struct named_query {
        std::string name;
        query_t q;
    };

    term V(uint64_t i) { term t; t.var = true; t.v = i; return t; }
    term C(uint64_t c) { term t; t.var = false; t.v = c; return t; }
    pattern P(term s, term p, term o) { pattern x; x.s = s; x.p = p; x.o = o; return x; }

    // ---------------- naive evaluator ----------------

    bool unify(const term &t, uint64_t val, std::vector<int64_t> &b, std::vector<uint64_t> &trail) {
        if (!t.var) return t.v == val;
        if (b[t.v] >= 0) return (uint64_t) b[t.v] == val;
        b[t.v] = (int64_t) val;
        trail.push_back(t.v);
        return true;
    }

    uint64_t naive_rec(const query_t &q, size_t i, const dataset_t &D, std::vector<int64_t> &b) {
        if (i == q.size()) return 1;
        uint64_t total = 0;
        for (size_t k = 0; k < D.size(); ++k) {
            std::vector<uint64_t> trail;
            bool ok = unify(q[i].s, std::get<0>(D[k]), b, trail) &&
                      unify(q[i].p, std::get<1>(D[k]), b, trail) &&
                      unify(q[i].o, std::get<2>(D[k]), b, trail);
            if (ok) total += naive_rec(q, i + 1, D, b);
            for (size_t j = 0; j < trail.size(); ++j) b[trail[j]] = -1;
        }
        return total;
    }

    uint64_t naive_count(const query_t &q, const dataset_t &D) {
        std::vector<int64_t> b(16, -1);
        return naive_rec(q, 0, D, b);
    }

    // ---------------- datasets ----------------

    std::mt19937 rng(42);

    uint64_t rnd(uint64_t n) { return rng() % n; }  // value in [0, n)

    dataset_t dedup(const dataset_t &D) {
        std::set<spo_triple> st(D.begin(), D.end());
        return dataset_t(st.begin(), st.end());
    }

    struct dataset_case {
        std::string name;
        dataset_t D;
    };

    dataset_t random_triples(size_t n, uint64_t ents, uint64_t preds) {
        dataset_t D;
        for (size_t i = 0; i < n; ++i)
            D.push_back(spo_triple(1 + rnd(ents), 1 + rnd(preds), 1 + rnd(ents)));
        return D;
    }

    std::vector<dataset_case> make_datasets() {
        std::vector<dataset_case> res;
        dataset_case c;
        // a) max ID only as subject
        c.name = "a-max-only-subject";
        c.D = random_triples(60, 20, 3);
        for (int i = 0; i < 4; ++i) c.D.push_back(spo_triple(30, 1 + rnd(3), 1 + rnd(20)));
        c.D = dedup(c.D); res.push_back(c);
        // b) max ID only as object
        c.name = "b-max-only-object";
        c.D = random_triples(60, 20, 3);
        for (int i = 0; i < 4; ++i) c.D.push_back(spo_triple(1 + rnd(20), 1 + rnd(3), 30));
        c.D = dedup(c.D); res.push_back(c);
        // c) max ID as subject and object
        c.name = "c-max-subject-and-object";
        c.D = random_triples(60, 20, 3);
        for (int i = 0; i < 3; ++i) c.D.push_back(spo_triple(30, 1 + rnd(3), 1 + rnd(20)));
        for (int i = 0; i < 3; ++i) c.D.push_back(spo_triple(1 + rnd(20), 1 + rnd(3), 30));
        c.D.push_back(spo_triple(30, 2, 30));
        c.D = dedup(c.D); res.push_back(c);
        // d) single triple (max ID as object / as subject)
        c.name = "d1-single-triple-max-object";
        c.D.clear(); c.D.push_back(spo_triple(3, 2, 5)); res.push_back(c);
        c.name = "d2-single-triple-max-subject";
        c.D.clear(); c.D.push_back(spo_triple(5, 1, 2)); res.push_back(c);
        c.name = "d3-single-triple-all-ones";
        c.D.clear(); c.D.push_back(spo_triple(1, 1, 1)); res.push_back(c);
        // e) all triples with the same subject (which is the max ID)
        c.name = "e-same-subject";
        c.D.clear();
        for (int i = 0; i < 25; ++i) c.D.push_back(spo_triple(20, 1 + rnd(3), 1 + rnd(15)));
        c.D = dedup(c.D); res.push_back(c);
        // f) random
        c.name = "f-random-400";
        c.D = dedup(random_triples(400, 60, 5)); res.push_back(c);
        return res;
    }

    // ---------------- queries ----------------

    bool has_triple(const dataset_t &D, uint64_t s, uint64_t p, uint64_t o) {
        for (size_t i = 0; i < D.size(); ++i)
            if (std::get<0>(D[i]) == s && std::get<1>(D[i]) == p && std::get<2>(D[i]) == o) return true;
        return false;
    }

    void add(std::vector<named_query> &qs, const std::string &name, const query_t &q) {
        named_query n; n.name = name; n.q = q; qs.push_back(n);
    }

    query_t Q(pattern a) { query_t q; q.push_back(a); return q; }
    query_t Q(pattern a, pattern b) { query_t q = Q(a); q.push_back(b); return q; }
    query_t Q(pattern a, pattern b, pattern c) { query_t q = Q(a, b); q.push_back(c); return q; }
    query_t Q(pattern a, pattern b, pattern c, pattern d) { query_t q = Q(a, b, c); q.push_back(d); return q; }

    // Queries built around the constants of triple t (an existing triple); tag identifies t.
    void queries_for(std::vector<named_query> &qs, const dataset_t &D, const spo_triple &t,
                     uint64_t max_id, const std::string &tag) {
        const uint64_t s = std::get<0>(t), p = std::get<1>(t), o = std::get<2>(t);
        const uint64_t p2 = (p % 3) + 1;
        add(qs, tag + " s ?p ?o", Q(P(C(s), V(0), V(1))));
        add(qs, tag + " ?s p ?o", Q(P(V(0), C(p), V(1))));
        add(qs, tag + " ?s ?p o", Q(P(V(0), V(1), C(o))));
        add(qs, tag + " s p ?o", Q(P(C(s), C(p), V(0))));
        add(qs, tag + " ?s p o", Q(P(V(0), C(p), C(o))));
        add(qs, tag + " s ?p o", Q(P(C(s), V(0), C(o))));
        add(qs, tag + " s p o", Q(P(C(s), C(p), C(o))));
        add(qs, tag + " path2", Q(P(V(0), C(p), V(1)), P(V(1), C(p), V(2))));
        add(qs, tag + " path2 s-const", Q(P(C(s), C(p), V(0)), P(V(0), C(p2), V(1))));
        add(qs, tag + " path2 o-const", Q(P(V(0), C(p), V(1)), P(V(1), C(p2), C(o))));
        add(qs, tag + " path2 mid-const", Q(P(V(0), C(p), C(o)), P(C(o), V(1), V(2))));
        add(qs, tag + " path4", Q(P(V(0), C(p), V(1)), P(V(1), C(p), V(2)),
                                  P(V(2), C(p), V(3)), P(V(3), C(p), V(4))));
        add(qs, tag + " path3 var-p", Q(P(V(0), V(1), V(2)), P(V(2), C(p), V(3)), P(V(3), V(4), C(o))));
        add(qs, tag + " triangle", Q(P(V(0), C(p), V(1)), P(V(1), C(p), V(2)), P(V(2), C(p), V(0))));
        add(qs, tag + " triangle mixed-p", Q(P(V(0), C(p), V(1)), P(V(1), C(p2), V(2)), P(V(2), V(3), V(0))));
        add(qs, tag + " star-s", Q(P(V(0), C(p), V(1)), P(V(0), C(p2), V(2)), P(V(0), V(3), V(4))));
        add(qs, tag + " star-s const", Q(P(C(s), V(0), V(1)), P(C(s), V(2), V(3))));
        add(qs, tag + " star-o", Q(P(V(0), C(p), V(2)), P(V(1), C(p2), V(2)), P(V(3), V(4), V(2))));
        add(qs, tag + " star-o const", Q(P(V(0), V(1), C(o)), P(V(2), V(3), C(o))));
        add(qs, tag + " var-p shared path", Q(P(V(0), V(1), V(2)), P(V(2), V(1), V(3))));
        add(qs, tag + " var-p shared cross", Q(P(V(0), V(1), V(2)), P(V(3), V(1), V(4))));
        add(qs, tag + " var-p shared star", Q(P(C(s), V(0), V(1)), P(V(2), V(0), C(o))));
        add(qs, tag + " all-vars", Q(P(V(0), V(1), V(2))));
        add(qs, tag + " dup pattern", Q(P(V(0), C(p), V(1)), P(V(0), C(p), V(1))));
        // empty queries: absent constant triples and non-joining patterns
        for (uint64_t x = 1; x <= max_id; ++x)
            if (!has_triple(D, s, p, x)) { add(qs, tag + " empty s p o'", Q(P(C(s), C(p), C(x)))); break; }
        for (uint64_t x = 1; x <= max_id; ++x)
            if (!has_triple(D, x, p, o)) { add(qs, tag + " empty s' p o", Q(P(C(x), C(p), V(0)), P(V(0), C(p), C(o)))); break; }
        // max ID as a constant in every position where it can appear
        add(qs, tag + " maxid as s", Q(P(C(max_id), V(0), V(1))));
        add(qs, tag + " maxid as o", Q(P(V(0), V(1), C(max_id))));
        add(qs, tag + " maxid as s,o", Q(P(C(max_id), V(0), C(max_id))));
        add(qs, tag + " maxid s + path", Q(P(C(max_id), V(0), V(1)), P(V(1), V(2), V(3))));
        add(qs, tag + " maxid o + path", Q(P(V(0), V(1), V(2)), P(V(2), V(3), C(max_id))));
    }

    std::vector<named_query> make_queries(const dataset_t &D) {
        std::vector<named_query> qs;
        uint64_t max_id = 0;
        spo_triple tm = D[0];
        for (size_t i = 0; i < D.size(); ++i) {
            uint64_t a = std::get<0>(D[i]), b = std::get<2>(D[i]);
            if (a > max_id) { max_id = a; tm = D[i]; }
            if (b > max_id) { max_id = b; tm = D[i]; }
        }
        queries_for(qs, D, D[0], max_id, "T0:");
        queries_for(qs, D, D[D.size() / 2], max_id, "Tmid:");
        queries_for(qs, D, tm, max_id, "Tmax:");
        return qs;
    }

    // ---------------- running the index ----------------

    std::vector<ring::triple_pattern> to_patterns(const query_t &q) {
        std::vector<ring::triple_pattern> v;
        for (size_t i = 0; i < q.size(); ++i) {
            ring::triple_pattern t;
            if (q[i].s.var) t.var_s(q[i].s.v); else t.const_s(q[i].s.v);
            if (q[i].p.var) t.var_p(q[i].p.v); else t.const_p(q[i].p.v);
            if (q[i].o.var) t.var_o(q[i].o.v); else t.const_o(q[i].o.v);
            v.push_back(t);
        }
        return v;
    }

    template<class Index, class VEO>
    uint64_t run_query(Index &idx, const query_t &q) {
        typedef ring::ltj_iterator<Index, uint8_t, uint64_t> it_t;
        typedef ring::ltj_algorithm<it_t, VEO> alg_t;
        typedef ::util::results_collector<typename alg_t::tuple_type> res_t;
        std::vector<ring::triple_pattern> pats = to_patterns(q);
        // results_collector allocates 2^20 vectors: reuse one per type and count the difference
        static res_t res;
        const uint64_t before = res.size();
        alg_t ltj(&pats, &idx);
        ltj.join_v2(res, 0, 0);
        return res.size() - before;
    }

    uint64_t g_checks = 0, g_fails = 0;

    template<class Index, class VEO>
    void check_queries(Index &idx, const std::vector<named_query> &qs, const std::vector<uint64_t> &expected,
                       const std::string &label) {
        uint64_t fails = 0;
        for (size_t i = 0; i < qs.size(); ++i) {
            uint64_t got = run_query<Index, VEO>(idx, qs[i].q);
            if (got != expected[i]) {
                ++fails;
                std::cout << "  FAIL " << label << " query '" << qs[i].name << "' expected "
                          << expected[i] << " got " << got << std::endl;
            }
        }
        ++g_checks;
        if (fails) ++g_fails;
        std::cout << (fails ? "FAIL " : "OK   ") << label << " (" << qs.size() << " queries, "
                  << fails << " mismatches)" << std::endl;
    }

    template<class Index, class VEO_ADAPTIVE, class VEO_SIMPLE>
    void check_index(const std::string &index_name, const dataset_case &dc,
                     const std::vector<named_query> &qs, const std::vector<uint64_t> &expected) {
        const std::string base = dc.name + " / " + index_name;
        dataset_t copy(dc.D);
        Index idx(copy);
        check_queries<Index, VEO_ADAPTIVE>(idx, qs, expected, base + " / adaptive");
        check_queries<Index, VEO_SIMPLE>(idx, qs, expected, base + " / simple");

        char tmpl[] = "/tmp/test-ring-muthu-XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd < 0) {
            std::cout << "FAIL " << base << " cannot create temporary file" << std::endl;
            ++g_checks; ++g_fails;
            return;
        }
        close(fd);
        std::string file(tmpl);
        sdsl::store_to_file(idx, file);
        Index loaded;
        sdsl::load_from_file(loaded, file);
        std::remove(file.c_str());
        check_queries<Index, VEO_ADAPTIVE>(loaded, qs, expected, base + " / adaptive / reloaded");
        check_queries<Index, VEO_SIMPLE>(loaded, qs, expected, base + " / simple / reloaded");
    }

}

int main() {
    std::cout << "seed = 42" << std::endl;
    typedef ring::ring<> ring_t;
    typedef ring::ring_muthu<> ring_muthu_t;
    typedef ring::c_ring_muthu c_ring_muthu_t;
    typedef ring::ring_sel_muthu ring_sel_muthu_t;

    std::vector<dataset_case> datasets = make_datasets();
    for (size_t d = 0; d < datasets.size(); ++d) {
        const dataset_case &dc = datasets[d];
        std::vector<named_query> qs = make_queries(dc.D);
        std::vector<uint64_t> expected;
        uint64_t empty = 0;
        for (size_t i = 0; i < qs.size(); ++i) {
            expected.push_back(naive_count(qs[i].q, dc.D));
            if (expected.back() == 0) ++empty;
        }
        std::cout << "== dataset " << dc.name << ": " << dc.D.size() << " triples, " << qs.size()
                  << " queries (" << empty << " empty)" << std::endl;

        {
            typedef ring::ltj_iterator<ring_t, uint8_t, uint64_t> it_t;
            check_index<ring_t,
                    ring::veo::veo_adaptive<it_t, ring::util::trait_size>,
                    ring::veo::veo_simple<it_t, ring::util::trait_size> >("ring", dc, qs, expected);
        }
        {
            typedef ring::ltj_iterator<ring_muthu_t, uint8_t, uint64_t> it_t;
            check_index<ring_muthu_t,
                    ring::veo::veo_adaptive<it_t, ring::util::trait_distinct>,
                    ring::veo::veo_simple<it_t, ring::util::trait_size> >("ring_muthu", dc, qs, expected);
        }
        {
            typedef ring::ltj_iterator<c_ring_muthu_t, uint8_t, uint64_t> it_t;
            check_index<c_ring_muthu_t,
                    ring::veo::veo_adaptive<it_t, ring::util::trait_distinct>,
                    ring::veo::veo_simple<it_t, ring::util::trait_size> >("c_ring_muthu", dc, qs, expected);
        }
        {
            typedef ring::ltj_iterator<ring_sel_muthu_t, uint8_t, uint64_t> it_t;
            check_index<ring_sel_muthu_t,
                    ring::veo::veo_adaptive<it_t, ring::util::trait_distinct>,
                    ring::veo::veo_simple<it_t, ring::util::trait_size> >("ring_sel_muthu", dc, qs, expected);
        }
    }

    std::cout << "SUMMARY: " << (g_checks - g_fails) << "/" << g_checks << " checks OK, "
              << g_fails << " FAIL" << std::endl;
    return g_fails ? 1 : 0;
}
