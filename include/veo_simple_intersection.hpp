/*
 * gao.hpp
 * Copyright (C) 2020 Author removed for double-blind evaluation
 *
 *
 * This is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef RING_VEO_SIMPLE_INTERSECTION_HPP
#define RING_VEO_SIMPLE_INTERSECTION_HPP

#include <ring.hpp>
#include <ltj_iterator.hpp>
#include <ltj_iterator_unidirectional.hpp>
#include <triple_pattern.hpp>
#include <unordered_map>
#include <vector>
#include <utils.hpp>
#include <unordered_set>

namespace ring {

    namespace veo {


        template<class ltj_iterator_t = ltj_iterator <ring<>, uint8_t, uint64_t>,
                 class veo_trait_t = util::trait_intersect<3>>
        class veo_simple_intersection {

        public:
            typedef ltj_iterator_t ltj_iter_type;
            typedef typename ltj_iter_type::var_type var_type;
            typedef uint64_t size_type;
            typedef typename ltj_iter_type::ring_type ring_type;
            typedef struct {
                var_type name;
                size_type weight;
                size_type n_triples;
                std::vector<size_type> params_weight;
                std::unordered_set<var_type> related;
            } info_var_type;

            typedef std::pair<size_type, var_type> pair_type;
            typedef veo_trait_t veo_trait_type;
            typedef std::priority_queue<pair_type, std::vector<pair_type>, greater<pair_type>> min_heap_type;
            typedef std::unordered_map<var_type, std::vector<ltj_iter_type*>> var_to_iterators_type;

        private:
            const std::vector<triple_pattern> *m_ptr_triple_patterns;
            const std::vector<ltj_iter_type> *m_ptr_iterators;
            ring_type *m_ptr_ring;
            std::vector<var_type> m_order;
            size_type m_index;
            size_type m_nolonely_size;

            void copy(const veo_simple_intersection &o) {
                m_ptr_triple_patterns = o.m_ptr_triple_patterns;
                m_ptr_iterators = o.m_ptr_iterators;
                m_ptr_ring = o.m_ptr_ring;
                m_order = o.m_order;
                m_index = o.m_index;
                m_nolonely_size = o.m_nolonely_size;
            }



            void var_to_vector(const var_type var, const size_type i, const state_type state,
                               std::unordered_map<var_type, size_type> &hash_table,
                               std::vector<info_var_type> &vec) {

                auto it = hash_table.find(var);
                if (it == hash_table.end()) {
                    info_var_type info;
                    info.name = var;
                    //info.weight = size;
                    info.params_weight = veo_trait_type::get(m_ptr_ring, m_ptr_iterators->at(i), state);
                    info.weight = 0;
                    for(const auto &p : info.params_weight){
                        info.weight += p;
                    }
                    info.n_triples = 1;
                    vec.emplace_back(info);
                    hash_table.insert({var, vec.size() - 1});
                } else {
                    info_var_type &info = vec[it->second];
                    ++info.n_triples;
                    info.weight = 0;
                    auto pw = veo_trait_type::get(m_ptr_ring, m_ptr_iterators->at(i), state);
                    for(size_type j = 0; j <pw.size(); ++j){
                        info.params_weight[j] = std::min(pw[j], info.params_weight[j]);
                        info.weight += info.params_weight[j];
                    }
                    /*if (info.weight > size) {
                        info.weight = size;
                    }*/
                }
            }

            void var_to_related(const var_type var, const var_type rel,
                                std::unordered_map<var_type, size_type> &hash_table,
                                std::vector<info_var_type> &vec) {

                auto pos_var = hash_table[var];
                vec[pos_var].related.insert(rel);
                auto pos_rel = hash_table[rel];
                vec[pos_rel].related.insert(var);
            }


            void fill_heap(const var_type var,
                           std::unordered_map<var_type, size_type> &hash_table,
                           std::vector<info_var_type> &vec,
                           std::vector<bool> &checked,
                           min_heap_type &heap) {

                auto pos_var = hash_table[var];
                for (const auto &e : vec[pos_var].related) {
                    auto pos_rel = hash_table[e];
                    if (!checked[pos_rel] && vec[pos_rel].n_triples > 1) {
                        heap.push({vec[pos_rel].weight, e});
                        checked[pos_rel] = true;
                    }
                }
            }

            struct compare_var_info {
                inline bool operator()(const info_var_type &linfo, const info_var_type &rinfo) {
                    if (linfo.n_triples > 1 && rinfo.n_triples == 1) {
                        return true;
                    }
                    if (linfo.n_triples == 1 && rinfo.n_triples > 1) {
                        return false;
                    }
                    return linfo.weight < rinfo.weight;
                }
            };

        public:

            veo_simple_intersection() = default;

            veo_simple_intersection(const std::vector<triple_pattern> *triple_patterns,
                       const std::vector<ltj_iter_type> *iterators,
                       const var_to_iterators_type *var_iterators,
                       ring_type *r) {
                m_ptr_triple_patterns = triple_patterns;
                m_ptr_iterators = iterators;
                m_ptr_ring = r;


                //1. Filling var_info with data about each variable
                //std::cout << "Filling... " << std::flush;
                std::vector<info_var_type> var_info;
                std::unordered_map<var_type, size_type> hash_table_position;
                size_type i = 0;
                for (const triple_pattern &triple_pattern : *m_ptr_triple_patterns) {
                    bool s = false, p = false, o = false;
                    var_type var_s, var_p, var_o;
                    size_type size;
                    if (triple_pattern.s_is_variable()) {
                        s = true;
                        var_s = (var_type) triple_pattern.term_s.value;
                        var_to_vector(var_s, i, state_type::s, hash_table_position, var_info);
                    }
                    if (triple_pattern.p_is_variable()) {
                        p = true;
                        var_p = (var_type) triple_pattern.term_p.value;
                        var_to_vector(var_p, i, state_type::p, hash_table_position, var_info);
                    }
                    if (triple_pattern.o_is_variable()) {
                        o = true;
                        var_o = (var_type) triple_pattern.term_o.value;
                        var_to_vector(var_o, i, state_type::o, hash_table_position, var_info);
                    }

                    if (s && p) {
                        var_to_related(var_s, var_p, hash_table_position, var_info);
                    }
                    if (s && o) {
                        var_to_related(var_s, var_o, hash_table_position, var_info);
                    }
                    if (p && o) {
                        var_to_related(var_p, var_o, hash_table_position, var_info);
                    }
                    ++i;
                }
                //std::cout << "Done. " << std::endl;

                //2. Sorting variables according to their weights.
                //std::cout << "Sorting... " << std::flush;
                std::sort(var_info.begin(), var_info.end(), compare_var_info());
                size_type lonely_start = var_info.size();
                for (i = 0; i < var_info.size(); ++i) {
                    hash_table_position[var_info[i].name] = i;
                    if (var_info[i].n_triples == 1 && i < lonely_start) {
                        lonely_start = i;
                    }
                }
                //std::cout << "Done. " << std::endl;
                m_nolonely_size = i;
                //3. Choosing the variables
                i = 0;
                //std::cout << "Choosing GAO ... " << std::flush;
                std::vector<bool> checked(var_info.size(), false);
                m_order.reserve(var_info.size());
                while (i < lonely_start) { //Related variables
                    if (!checked[i]) {
                        m_order.push_back(var_info[i].name); //Adding var to gao
                        checked[i] = true;
                        min_heap_type heap; //Stores the related variables that are related with the chosen ones
                        auto var_name = var_info[i].name;
                        fill_heap(var_name, hash_table_position, var_info, checked, heap);
                        while (!heap.empty()) {
                            var_name = heap.top().second;
                            heap.pop();
                            m_order.push_back(var_name);
                            fill_heap(var_name, hash_table_position, var_info, checked, heap);
                        }
                    }
                    ++i;
                }
                while (i < var_info.size()) { //Lonely variables
                    m_order.push_back(var_info[i].name); //Adding var to gao
                    ++i;
                }
                m_index = 0;
                //std::cout << "Done. " << std::endl;
            }

            //! Copy constructor
            veo_simple_intersection(const veo_simple_intersection &o) {
                copy(o);
            }

            //! Move constructor
            veo_simple_intersection(veo_simple_intersection &&o) {
                *this = std::move(o);
            }

            //! Copy Operator=
            veo_simple_intersection &operator=(const veo_simple_intersection &o) {
                if (this != &o) {
                    copy(o);
                }
                return *this;
            }

            //! Move Operator=
            veo_simple_intersection &operator=(veo_simple_intersection &&o) {
                if (this != &o) {
                    m_ptr_triple_patterns = std::move(o.m_ptr_triple_patterns);
                    m_ptr_iterators = std::move(o.m_ptr_iterators);
                    m_ptr_ring = o.m_ptr_ring;
                    m_order = std::move(o.m_order);
                    m_index = o.m_index;
                    m_nolonely_size = o.m_nolonely_size;
                }
                return *this;
            }

            void swap(veo_simple_intersection &o) {
                std::swap(m_ptr_triple_patterns, o.m_ptr_triple_patterns);
                std::swap(m_ptr_iterators, o.m_ptr_iterators);
                std::swap(m_ptr_ring, o.m_ptr_ring);
                std::swap(m_order, o.m_order);
                std::swap(m_index, o.m_index);
                std::swap(m_nolonely_size, o.m_nolonely_size);
            }

            inline var_type next() {
                ++m_index;
                return m_order[m_index-1];
            }



            inline void down() {
                //++m_index;
            };

            inline void up() {
               // --m_index;
            };

            inline void done() {
                --m_index;
            };

            inline size_type size() {
                return m_order.size();
            }

            inline size_type nolonely_size() {
                return m_nolonely_size;
            }
        };
    };
}

#endif //RING_GAO_SIZE_HPP
