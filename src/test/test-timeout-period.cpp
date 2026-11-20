/*
 * test-timeout-period.cpp
 *
 * Tests for the timeout check of ltj_algorithm and ltj_algorithm_unidirectional (join, join_v2, join_ramas):
 * the clock is read once every timeout_check_period calls to search, so results must not depend on it,
 * and the timeout must still fire. Self-contained: datasets are generated in memory with a fixed seed.
 * Returns != 0 if any check fails.
 */

#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <ring.hpp>
#include <uring.hpp>
#include <triple_pattern.hpp>
#include <ltj_algorithm.hpp>
#include <ltj_algorithm_unidirectional.hpp>
#include <ltj_iterator.hpp>
#include <ltj_iterator_unidirectional.hpp>
#include <results_collector.hpp>

using namespace std;

static const uint32_t SEED = 42;
static uint64_t n_ok = 0, n_fail = 0;

typedef std::vector<ring::triple_pattern> query_type;

static void check(bool cond, const std::string &what) {
    if (cond) {
        ++n_ok;
        std::cout << "OK   " << what << std::endl;
    } else {
        ++n_fail;
        std::cout << "FAIL " << what << std::endl;
    }
}

//Random dataset with predicates 1..n_preds (all present) and entities 1..n_ent
static std::vector<spo_triple> gen_dataset(std::mt19937 &gen, uint32_t n_triples, uint32_t n_ent,
                                           uint32_t n_preds) {
    std::set<spo_triple> S;
    std::uniform_int_distribution<uint32_t> ent(1, n_ent), pred(1, n_preds);
    for (uint32_t p = 1; p <= n_preds; ++p) { //every predicate appears
        uint32_t s = ent(gen), o = ent(gen);
        S.insert(spo_triple(s, p, o));
    }
    while (S.size() < n_triples) {
        uint32_t s = ent(gen), p = pred(gen), o = ent(gen);
        S.insert(spo_triple(s, p, o));
    }
    return std::vector<spo_triple>(S.begin(), S.end());
}

//Minimal parser: "?x 1 ?y . ?y 2 ?z", variables numbered by first appearance
static query_type get_query(const std::string &str) {
    std::unordered_map<std::string, uint8_t> vars;
    query_type q;
    std::stringstream ss(str);
    std::string tp;
    while (std::getline(ss, tp, '.')) {
        std::stringstream st(tp);
        std::string t[3];
        st >> t[0] >> t[1] >> t[2];
        ring::triple_pattern p;
        for (int k = 0; k < 3; ++k) {
            const bool isv = (t[k][0] == '?');
            uint64_t v = 0;
            if (isv) {
                auto it = vars.find(t[k]);
                if (it == vars.end()) {
                    v = vars.size();
                    vars.insert({t[k], (uint8_t) v});
                } else {
                    v = it->second;
                }
            } else {
                v = std::stoull(t[k]);
            }
            if (k == 0) { if (isv) p.var_s(v); else p.const_s(v); }
            else if (k == 1) { if (isv) p.var_p(v); else p.const_p(v); }
            else { if (isv) p.var_o(v); else p.const_o(v); }
        }
        q.push_back(p);
    }
    return q;
}

//Naive evaluator: nested loops over the dataset, one pattern at a time
static uint64_t naive_rec(const query_type &q, size_t i, const std::vector<spo_triple> &D,
                          std::vector<uint64_t> &bind) {
    if (i == q.size()) return 1;
    uint64_t cnt = 0;
    const ring::triple_pattern &tp = q[i];
    for (const auto &t : D) {
        const bool isv[3] = {tp.s_is_variable(), tp.p_is_variable(), tp.o_is_variable()};
        const uint64_t val[3] = {tp.term_s.value, tp.term_p.value, tp.term_o.value};
        const uint64_t tv[3] = {std::get<0>(t), std::get<1>(t), std::get<2>(t)};
        uint64_t assigned[3];
        int n_assigned = 0;
        bool ok = true;
        for (int k = 0; k < 3 && ok; ++k) {
            if (!isv[k]) {
                ok = (tv[k] == val[k]);
            } else if (bind[val[k]] == 0) {
                bind[val[k]] = tv[k];
                assigned[n_assigned++] = val[k];
            } else {
                ok = (bind[val[k]] == tv[k]);
            }
        }
        if (ok) cnt += naive_rec(q, i + 1, D, bind);
        for (int a = 0; a < n_assigned; ++a) bind[assigned[a]] = 0;
    }
    return cnt;
}

static uint64_t naive_count(const query_type &q, const std::vector<spo_triple> &D) {
    std::vector<uint64_t> bind(256, 0);
    return naive_rec(q, 0, D, bind);
}

//No variable is repeated inside a pattern (project precondition)
static const char *QUERIES[] = {
    "?x 1 ?y . ?y 2 ?z",                         //path
    "?x 1 ?y . ?y 2 ?z . ?z 3 ?w",               //long path
    "?x 1 ?y . ?y 2 ?z . ?z 3 ?x",               //triangle
    "?x 1 ?y . ?x 2 ?z . ?x 3 ?w",               //star (subject)
    "?x 1 ?y . ?z 2 ?y . ?w 3 ?y",               //star (object)
    "5 ?p ?o",                                   //constant subject
    "?s 2 7",                                    //constant predicate and object
    "3 1 ?o",                                    //constant subject and predicate
    "?x ?p ?y . ?y ?q ?z",                       //variable predicates
    "?x ?p ?y . ?z ?p ?y",                       //shared variable predicate
    "?x 1 ?y . ?y 2 ?x",                         //cycle of length 2
    "?x 2 61",                                   //empty (61 is only a subject)
    "?x 1 ?y . ?y 2 61"                          //empty
};

static const uint64_t LIMITS[] = {0, 1, 1000};
static const uint64_t TIMEOUTS[] = {600, 0};

//variant: 0 = join, 1 = join_v2, 2 = join_ramas (only ltj_algorithm)
template<class Iterator, class Veo>
static uint64_t run_query(ring::ltj_algorithm<Iterator, Veo> &ltj, int variant, uint64_t limit, uint64_t timeout) {
    typedef ring::ltj_algorithm<Iterator, Veo> algorithm_type;
    if (variant == 2) {
        std::vector<typename algorithm_type::tuple_type> res;
        uint64_t ramas = 0;
        ltj.join_ramas(res, ramas, limit, timeout);
        return (uint64_t) res.size();
    }
    ::util::results_collector<typename algorithm_type::tuple_type> res;
    if (variant == 0) ltj.join(res, limit, timeout);
    else ltj.join_v2(res, limit, timeout);
    return (uint64_t) res.size();
}

//ltj_algorithm_unidirectional only has join
template<class Iterator, class Veo>
static uint64_t run_query(ring::ltj_algorithm_unidirectional<Iterator, Veo> &ltj, int, uint64_t limit,
                          uint64_t timeout) {
    ::util::results_collector<typename ring::ltj_algorithm_unidirectional<Iterator, Veo>::tuple_type> res;
    ltj.join(res, limit, timeout);
    return (uint64_t) res.size();
}

template<class Index, class Algorithm>
static void run_index(const std::string &name, const std::vector<spo_triple> &data0, int n_variants) {
    std::vector<spo_triple> D = data0; //the index constructor reorders D
    Index index(D);
    static const char *VNAMES[] = {"/join", "/join_v2", "/join_ramas"};

    for (int variant = 0; variant < n_variants; ++variant) {
        const std::string vname = name + VNAMES[variant];
        bool same = true, nonzero = false, zero = false;
        for (const char *qs : QUERIES) {
            query_type q = get_query(qs);
            const uint64_t expected = naive_count(q, data0);
            if (expected > 0) nonzero = true; else zero = true;
            for (uint64_t limit : LIMITS) {
                const uint64_t exp_l = (limit > 0 && expected > limit) ? limit : expected;
                for (uint64_t timeout : TIMEOUTS) {
                    Algorithm ltj(&q, &index);
                    const uint64_t r = run_query(ltj, variant, limit, timeout);
                    if (r != exp_l) {
                        same = false;
                        std::cout << "  query '" << qs << "' limit " << limit << " timeout " << timeout
                                  << ": got " << r << " expected " << exp_l << std::endl;
                    }
                }
            }
        }
        check(nonzero && zero, vname + ": test queries include empty and non-empty results");
        check(same, vname + ": same results as naive with timeout 600 and 0, limit 0/1/1000");
    }
}

//The timeout still fires on a huge result
template<class Index, class Algorithm>
static void run_timeout(const std::string &name, const std::vector<spo_triple> &data0, int n_variants) {
    std::vector<spo_triple> D = data0;
    Index index(D);
    query_type q = get_query("?a ?p ?b . ?c ?q ?d . ?e ?r ?f");
    for (int variant = 0; variant < n_variants; ++variant) {
        Algorithm ltj(&q, &index);
        auto t0 = std::chrono::steady_clock::now();
        const uint64_t n = run_query(ltj, variant, 0, 1);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        check(secs < 4.0 && n > 0,
              name + (variant == 0 ? "/join" : "/join_v2") + " timeout 1s: "
              + std::to_string(secs) + " s, " + std::to_string(n) + " results");
    }
}

int main() {
    std::cout << "seed = " << SEED << std::endl;
    std::mt19937 gen(SEED);

    //Small dataset: entities 1..60, predicates 1..5; 61 is only a subject
    std::vector<spo_triple> D = gen_dataset(gen, 400, 60, 5);
    {
        std::set<spo_triple> S(D.begin(), D.end());
        S.insert(spo_triple(61, 1, 1));
        D.assign(S.begin(), S.end());
    }
    //Big dataset for the timeout test
    std::vector<spo_triple> Dbig = gen_dataset(gen, 3000, 200, 5);

    //Bidirectional ring
    #define RING_TESTS(NAME, IDX) { \
        typedef ring::ltj_iterator<IDX, uint8_t, uint64_t> it; \
        typedef ring::ltj_algorithm<it, ring::veo::veo_adaptive<it, ring::util::trait_size>> alg_a; \
        typedef ring::ltj_algorithm<it, ring::veo::veo_simple<it, ring::util::trait_size>> alg_s; \
        run_index<IDX, alg_a>(NAME "/adaptive", D, 3); \
        run_index<IDX, alg_s>(NAME "/simple", D, 3); }
    RING_TESTS("ring", ring::ring<>)
    RING_TESTS("c_ring", ring::c_ring)
    RING_TESTS("ring_sel", ring::ring_sel)

    //Unidirectional ring (as in query-index-uring.cpp)
    #define URING_TESTS(NAME, IDX) { \
        typedef ring::ltj_iterator_unidirectional<IDX, uint8_t, uint64_t> it; \
        typedef ring::ltj_algorithm_unidirectional<it, ring::veo::veo_adaptive<it, ring::util::trait_size>> alg_a; \
        typedef ring::ltj_algorithm_unidirectional<it, ring::veo::veo_simple<it, ring::util::trait_size>> alg_s; \
        run_index<IDX, alg_a>(NAME "/adaptive", D, 1); \
        run_index<IDX, alg_s>(NAME "/simple", D, 1); }
    URING_TESTS("uring", ring::uring<>)
    URING_TESTS("c_uring", ring::c_uring)

    //Timeout on a huge result
    {
        typedef ring::ltj_iterator<ring::ring<>, uint8_t, uint64_t> it;
        typedef ring::ltj_algorithm<it, ring::veo::veo_adaptive<it, ring::util::trait_size>> alg_a;
        typedef ring::ltj_algorithm<it, ring::veo::veo_simple<it, ring::util::trait_size>> alg_s;
        run_timeout<ring::ring<>, alg_a>("ring/adaptive", Dbig, 2);
        run_timeout<ring::ring<>, alg_s>("ring/simple", Dbig, 2);
    }
    {
        typedef ring::ltj_iterator_unidirectional<ring::uring<>, uint8_t, uint64_t> it;
        typedef ring::ltj_algorithm_unidirectional<it, ring::veo::veo_adaptive<it, ring::util::trait_size>> alg_a;
        typedef ring::ltj_algorithm_unidirectional<it, ring::veo::veo_simple<it, ring::util::trait_size>> alg_s;
        run_timeout<ring::uring<>, alg_a>("uring/adaptive", Dbig, 1);
        run_timeout<ring::uring<>, alg_s>("uring/simple", Dbig, 1);
    }

    std::cout << n_ok << " checks OK, " << n_fail << " FAIL" << std::endl;
    return n_fail == 0 ? 0 : 1;
}
